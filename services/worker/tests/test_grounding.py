"""Grounding the analysis's 2D boxes on the floor: rays, doors and furniture.

Each scene is built analytically (see scene_analysis_fixtures): true boxes are projected
through known cameras, so the recovered doors and furniture can be checked against the truth.
The cameras walk frontally, about 4 m from the object, which is where a box's bottom edge
reads its floor position and its width reliably.
"""

from __future__ import annotations

import numpy as np
import pytest
from scene_analysis_fixtures import (
    agent_object,
    analysis_artifact,
    box_corners,
    detection,
    door_detection,
    keyframe,
)

from andarilho_worker.geometry.grounding import (
    floor_hit,
    floor_objects,
    ground_doors,
    ground_objects,
    views_from_analysis,
)

CENTRE = (4.0, 3.0)  # where the furniture stands, in the canonical world
EYE = 1.5


def frontal_cameras(distance: float = 4.0, lateral: tuple[float, ...] = (-0.5, 0.0, 0.5)):
    """Cameras in front of the object (on +z) at eye height, looking at its middle."""
    return [(CENTRE[0] + dx, EYE, CENTRE[1] + distance) for dx in lateral]


def furniture_scene(agent_id, category, size, *, yaw=0.0, cameras=None, agent_size=None):
    """Frames that see one floor object from the cameras, with the agent's size as prior."""
    cameras = cameras or frontal_cameras()
    keyframes = [keyframe(i + 1, c, (CENTRE[0], 0.45, CENTRE[1])) for i, c in enumerate(cameras)]
    frames = []
    for record in keyframes:
        found = detection(record, box_corners(CENTRE, yaw, size, 0.0))
        objects = [] if found is None else [{"id": agent_id, **found}]
        frames.append({"frame": record["frame"], "room": "r1", "doors": [], "objects": objects})
    agent = agent_object(agent_id, category, agent_size or size)
    return analysis_artifact(keyframes, frames, [agent])


def _grounded(analysis):
    got = ground_objects(analysis, views_from_analysis(analysis))
    assert len(got) == 1
    return got[0]


def test_views_put_the_camera_where_the_pose_says_and_look_at_the_target():
    target = (4.0, 0.45, 3.0)
    record = keyframe(7, (1.0, 1.5, 0.5), target)
    view = views_from_analysis({"keyframes": [record]})[7]
    assert view.centre() == pytest.approx([1.0, 1.5, 0.5])
    # The image centre looks straight at the target.
    direction = view.ray(0.5, 0.5)
    expected = np.array(target) - view.centre()
    assert direction == pytest.approx(expected / np.linalg.norm(expected), abs=1e-9)


def test_floor_hit_is_the_point_under_the_ray_and_none_above_the_horizon():
    # A camera 1.5 m up, looking level: the image's middle row is the horizon.
    level = keyframe(1, (0.0, 1.5, 0.0), (0.0, 1.5, 5.0))
    view = views_from_analysis({"keyframes": [level]})[1]
    # 0.9 of the way down the image is 1.5 m / 0.46 = 3.25 m out along the centre column.
    below = floor_hit(view, 0.5, 0.9)
    assert below == pytest.approx([0.0, 3.25], abs=0.02)
    assert floor_hit(view, 0.5, 0.2) is None  # above the horizon: the ray goes up
    assert floor_hit(view, 0.5, 0.5) is None  # exactly level: it never reaches the floor


def test_floor_hit_beyond_the_range_is_none():
    level = keyframe(1, (0.0, 1.5, 0.0), (0.0, 1.5, 5.0))
    view = views_from_analysis({"keyframes": [level]})[1]
    # Just below the horizon the floor is many metres away.
    assert floor_hit(view, 0.5, 0.5 + 1e-3) is None


def test_sofa_is_placed_and_sized_from_frontal_frames():
    agent_size = (2.0, 0.9, 0.9)
    got = _grounded(furniture_scene("o1", "sofa", agent_size))
    assert np.hypot(got.position[0] - CENTRE[0], got.position[1] - CENTRE[1]) < 0.15
    assert got.size[0] == pytest.approx(2.0, rel=0.2)
    assert got.size[2] == pytest.approx(0.9, rel=0.2)
    assert got.size[1] == pytest.approx(0.9, rel=0.15)
    assert got.category == "sofa"
    assert got.solid is True
    assert got.parts == agent_object("o1", "sofa", agent_size)["parts"]


def test_a_tall_wardrobe_above_eye_level_reads_its_front_top_edge():
    size = (1.0, 2.0, 0.6)  # its top edge is above the camera
    got = _grounded(furniture_scene("o1", "wardrobe", size))
    assert np.hypot(got.position[0] - CENTRE[0], got.position[1] - CENTRE[1]) < 0.15
    assert got.size[1] == pytest.approx(2.0, rel=0.15)
    assert got.size[0] == pytest.approx(1.0, rel=0.2)


def test_a_low_table_reads_its_back_top_edge():
    size = (1.2, 0.75, 0.6)  # its top edge is below the camera, so the back edge shows
    got = _grounded(furniture_scene("o1", "table", size))
    assert np.hypot(got.position[0] - CENTRE[0], got.position[1] - CENTRE[1]) < 0.15
    assert got.size[1] == pytest.approx(0.75, rel=0.15)


def test_the_agent_size_is_a_prior_that_measurements_may_move_within_a_factor_of_two():
    size = (2.0, 0.9, 0.9)
    # The agent says the sofa is half as wide as it is: the measured 2 m is clamped to 2 w.
    got = _grounded(furniture_scene("o1", "sofa", size, agent_size=(0.5, 0.9, 0.9)))
    assert got.size[0] == pytest.approx(1.0)
    # The depth keeps the agent's aspect ratio (d * width / w = 1.8 m), clamped to 1.5 d.
    assert got.size[2] == pytest.approx(1.35)
    assert got.size[1] == pytest.approx(0.9, rel=0.15)


def test_confidence_grows_with_the_number_of_frames_up_to_a_cap():
    one = _grounded(furniture_scene("o1", "sofa", (2.0, 0.9, 0.9), cameras=frontal_cameras()[:1]))
    three = _grounded(furniture_scene("o1", "sofa", (2.0, 0.9, 0.9)))
    assert one.detections == 1 and one.confidence == pytest.approx(0.45)
    assert three.detections == 3 and three.confidence == pytest.approx(0.75)
    many = frontal_cameras(lateral=(-0.6, -0.3, 0.0, 0.3, 0.6))
    assert _grounded(furniture_scene("o1", "sofa", (2.0, 0.9, 0.9), cameras=many)).confidence == (
        pytest.approx(0.9)
    )


def test_an_object_whose_bottom_is_cut_off_in_every_frame_is_dropped():
    analysis = furniture_scene("o1", "sofa", (2.0, 0.9, 0.9))
    for entry in analysis["reply"]["frames"]:
        for item in entry["objects"]:
            item["bottomVisible"] = False
    assert ground_objects(analysis, views_from_analysis(analysis)) == []


def test_a_box_whose_bottom_edge_is_above_the_horizon_is_dropped():
    level = keyframe(1, (0.0, 1.5, 0.0), (0.0, 1.5, 5.0))
    analysis = analysis_artifact(
        [level],
        [
            {
                "frame": 1,
                "room": "r1",
                "doors": [],
                "objects": [{"id": "o1", "box": [0.4, 0.2, 0.6, 0.45], "bottomVisible": True}],
            }
        ],  # fmt: skip
        [agent_object("o1", "sofa", (2.0, 0.9, 0.9))],
    )
    assert ground_objects(analysis, views_from_analysis(analysis)) == []


def test_an_object_farther_than_ten_metres_is_dropped():
    level = keyframe(1, (0.0, 1.5, 0.0), (0.0, 1.5, 5.0))
    # Level camera: a floor point D metres out sits at 0.5 + 1.3 / D of the image height,
    # so this bottom edge is on the floor 11 m away, beyond the usable 10 m.
    y_bottom = 0.5 + 1.3 / 11.0
    analysis = analysis_artifact(
        [level],
        [
            {
                "frame": 1,
                "room": "r1",
                "doors": [],
                "objects": [
                    {
                        "id": "o1",
                        "box": [0.4, y_bottom - 0.05, 0.6, y_bottom],
                        "bottomVisible": True,
                    }
                ],
            }
        ],  # fmt: skip
        [agent_object("o1", "sofa", (2.0, 0.9, 0.9))],
    )
    assert ground_objects(analysis, views_from_analysis(analysis)) == []


def test_wall_and_ceiling_items_are_not_grounded_on_the_floor():
    analysis = furniture_scene("o1", "shelf", (1.0, 0.3, 0.2))
    analysis["reply"]["objects"][0]["mount"] = "wall"
    assert floor_objects(analysis) == {}
    assert ground_objects(analysis, views_from_analysis(analysis)) == []


def test_a_detection_in_a_frame_without_a_pose_is_ignored():
    analysis = furniture_scene("o1", "sofa", (2.0, 0.9, 0.9))
    analysis["keyframes"] = analysis["keyframes"][1:]
    got = _grounded(analysis)  # the remaining two frames still ground it
    assert got.detections == 2


# --------------------------------------------------------------------------
# Doors
# --------------------------------------------------------------------------


def _door_scene(width=0.9, height=2.05, *, state="closed", leads_to=None, yaw=0.0):
    """A door in a wall at z = 7, seen frontally from the room side."""
    # Five metres out, so the camera is nearly level with the door's middle: the pitch that
    # makes the top corners read closer than the bottom ones stays small.
    wall_centre = (3.0, 7.0)
    cameras = [(2.8, EYE, 2.0), (3.0, EYE, 2.0), (3.2, EYE, 2.0)]
    keyframes, frames = [], []
    for i, c in enumerate(cameras):
        record = keyframe(i + 1, c, (wall_centre[0], 1.0, wall_centre[1]))
        keyframes.append(record)
        found = door_detection(
            record, wall_centre, yaw, width, height, state=state, leads_to=leads_to
        )
        frames.append({"frame": i + 1, "room": "r1", "objects": [],
                       "doors": [] if found is None else [found]})  # fmt: skip
    return analysis_artifact(keyframes, frames, [])


def test_a_door_is_grounded_at_its_midpoint_with_its_width_and_top():
    analysis = _door_scene(width=0.9, height=2.05, state="open", leads_to="r2")
    doors = ground_doors(analysis, views_from_analysis(analysis))
    assert len(doors) == 3
    for door in doors:
        assert door.position[0] == pytest.approx(3.0, abs=0.1)
        assert door.position[1] == pytest.approx(7.0, abs=0.1)
        assert door.width == pytest.approx(0.9, abs=0.1)
        assert door.top == pytest.approx(2.05, abs=0.1)
        assert door.state == "open"
        assert door.leads_to == "r2"
        # The jambs run along the wall: the direction is along x, up to sign and a few degrees.
        assert abs(door.direction[0]) > 0.99


def test_a_door_cut_off_at_the_bottom_of_the_frame_is_dropped():
    analysis = _door_scene()
    for entry in analysis["reply"]["frames"]:
        for door in entry["doors"]:
            door["box"][3] = 0.99  # the bottom edge is out of frame
    assert ground_doors(analysis, views_from_analysis(analysis)) == []


def test_a_door_top_out_of_frame_falls_back_to_the_standard_height():
    analysis = _door_scene()
    for entry in analysis["reply"]["frames"]:
        for door in entry["doors"]:
            door["box"][1] = 0.0  # the top edge is out of frame
    doors = ground_doors(analysis, views_from_analysis(analysis))
    assert doors and all(door.top == pytest.approx(2.05) for door in doors)


def test_a_door_whose_bottom_corners_miss_the_floor_is_dropped():
    level = keyframe(1, (3.0, 1.5, 4.0), (3.0, 1.5, 7.0))  # looking level at the wall
    analysis = analysis_artifact(
        [level],
        [
            {
                "frame": 1,
                "room": "r1",
                "objects": [],
                "doors": [{"box": [0.4, 0.2, 0.6, 0.45], "state": "closed", "leadsTo": None}],
            }
        ],  # fmt: skip
        [],
    )
    assert ground_doors(analysis, views_from_analysis(analysis)) == []
