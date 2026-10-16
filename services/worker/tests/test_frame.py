"""Up, floor, estimated scale and spawn from the camera path (geometry.frame)."""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest
from colmap_fixtures import (
    COLMAP_SCALE,
    FLOOR,
    UP,
    apply_affine,
    axis_rotation,
    camera_in_colmap,
    cameras_for,
    colmap_transform,
    join_points,
    l_walk,
    make_room,
    similarity_matrix_problem,
    to_sparse,
    with_trust,
)

from andarilho_worker.geometry.colmap import read_camera_poses
from andarilho_worker.geometry.frame import (
    CAMERA_HEIGHT_METERS,
    PATH_MARGIN_METERS,
    CameraPath,
    _nearest_neighbours,
    _up_from_cameras,
    estimate_frame,
    frame_from_sparse,
    reliable_points,
    rotation_to_z,
)


def _hallway(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A level hallway in true meters (Z up): floor, two walls, a ceiling, a walk.

    The walker goes +X at 1.5 m with the phone pitched 25 degrees down.
    """
    n = 6000
    floor = np.column_stack([rng.uniform(-1, 6, n), rng.uniform(-1, 1, n), np.zeros(n)])
    side = rng.choice([-1.0, 1.0], n)
    walls = np.column_stack([rng.uniform(-1, 6, n), side, rng.uniform(0, 2.5, n)])
    ceiling = np.column_stack([rng.uniform(-1, 6, n // 3), rng.uniform(-1, 1, n // 3),
                               np.full(n // 3, 2.5)])  # fmt: skip
    floaters = rng.normal([2, 0, -3], 1.0, (400, 3))  # junk well below the floor
    points = np.vstack([floor, walls, ceiling, floaters])
    centers = np.column_stack([np.linspace(0, 5, 30), np.zeros(30), np.full(30, 1.5)])
    pitch = math.radians(25)
    forwards = np.tile([math.cos(pitch), 0.0, -math.sin(pitch)], (30, 1))
    return points + rng.normal(0, 0.005, points.shape), centers, forwards


@pytest.mark.parametrize("tilt", [0.0, 12.0, 25.0])
def test_a_tilted_unscaled_capture_is_levelled_and_put_at_walking_height(tilt):
    rng = np.random.default_rng(1)
    points, centers, forwards = _hallway(rng)
    # What the trainer hands over: an arbitrary scale, a tilted up axis, an offset.
    tilt_r = axis_rotation(np.array([0.3, 1.0, 0.0]), tilt)
    scale, offset = 0.21, np.array([0.4, -0.2, 0.3])
    export = points @ tilt_r.T * scale + offset
    path = CameraPath(centers @ tilt_r.T * scale + offset, forwards @ tilt_r.T)

    est = estimate_frame(export, path, seed=3)

    assert est.method == "floor_plane"
    assert similarity_matrix_problem(est.matrix) is None
    assert est.tilt_degrees == pytest.approx(tilt, abs=1.0)
    assert est.meters_per_unit == pytest.approx(1 / scale, rel=0.03)
    world_floor = apply_affine(est.matrix, points[:6000] @ tilt_r.T * scale + offset)
    assert np.abs(world_floor[:, 1]).max() < 0.05  # the floor is level at y = 0
    world_cams = apply_affine(est.matrix, path.centers)
    assert world_cams[:, 1] == pytest.approx(CAMERA_HEIGHT_METERS, abs=0.05)
    # The spawn is where the walk started, on the floor, facing along the walk.
    assert est.spawn_feet == pytest.approx([world_cams[0, 0], 0.0, world_cams[0, 2]], abs=1e-6)
    walk = world_cams[-1] - world_cams[0]
    facing = np.array([-math.sin(est.spawn_yaw), 0.0, -math.cos(est.spawn_yaw)])
    assert facing @ walk / np.linalg.norm(walk) > 0.99
    # The proposed floor holds the whole walk with room around it.
    assert est.min_xz[0] < world_cams[:, 0].min() - 0.5
    assert est.max_xz[0] > world_cams[:, 0].max() + 0.5
    assert est.height == pytest.approx(2.5, abs=0.15)
    assert est.warnings == []


def test_without_a_floor_it_keeps_the_trainer_up_and_says_so():
    rng = np.random.default_rng(2)
    walls = np.column_stack([rng.uniform(0, 5, 4000), rng.choice([-1.0, 1.0], 4000),
                             rng.uniform(0, 2.5, 4000)])  # fmt: skip
    centers = np.column_stack([np.linspace(0, 5, 20), np.zeros(20), np.full(20, 1.5)])
    path = CameraPath(centers, np.tile([1.0, 0, 0], (20, 1)))
    est = estimate_frame(walls, path, seed=0)
    assert est.method == "trainer_up"
    assert est.tilt_degrees == 0.0
    assert similarity_matrix_problem(est.matrix) is None
    assert any("floor" in w for w in est.warnings)


def test_a_bent_walk_is_flagged_for_review():
    rng = np.random.default_rng(4)
    points, centers, forwards = _hallway(rng)
    centers = centers.copy()
    centers[:, 2] = np.linspace(0.8, 2.6, len(centers))  # the path climbs 1.8 m
    est = estimate_frame(points, CameraPath(centers, forwards), seed=0)
    assert any("varies" in w for w in est.warnings)


def test_rotation_to_z_is_proper():
    for normal in ([0, 0, 1.0], [0.2, -0.1, 0.97], [0, 0, -1.0]):
        n = np.array(normal) / np.linalg.norm(normal)
        r = rotation_to_z(n)
        assert r @ n == pytest.approx([0, 0, 1.0], abs=1e-9)
        assert np.linalg.det(r) == pytest.approx(1.0)


def test_read_camera_poses_returns_centres_and_viewing_directions(tmp_path):
    # Camera 2 is rotated 90 degrees about +Y (q = cos45, 0, sin45, 0), t = (1, 2, 3).
    half = math.sqrt(0.5)
    images = [
        (2, (half, 0.0, half, 0.0), (1.0, 2.0, 3.0), "00/frame_00002.jpg"),
        (1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -4.0), "00/frame_00001.jpg"),
    ]
    blob = struct.pack("<Q", len(images))
    for image_id, q, t, name in images:
        blob += struct.pack("<I7dI", image_id, *q, *t, 1) + name.encode() + bytes(1)
        blob += struct.pack("<Q", 1) + struct.pack("<ddQ", 10.0, 20.0, 7)
    path = tmp_path / "images.bin"
    path.write_bytes(blob)
    first, second = read_camera_poses(path)
    assert first["name"] == "00/frame_00001.jpg"
    assert first["center"] == pytest.approx([0, 0, 4.0])
    assert first["forward"] == pytest.approx([0, 0, 1.0])
    # R = rot_y(90): centre = -R^T t, forward = R^T (0, 0, 1) = (-1, 0, 0).
    assert second["center"] == pytest.approx([3.0, -2.0, -1.0])
    assert second["forward"] == pytest.approx([-1.0, 0, 0], abs=1e-12)


def _axis_residual_degrees(xz: np.ndarray) -> float:
    """How far the principal direction of an XZ point cloud is from the nearest X or Z axis."""
    centred = xz - xz.mean(axis=0)
    _, vectors = np.linalg.eigh(centred.T @ centred)
    dx, dz = np.abs(vectors[:, -1])
    angle = math.degrees(math.atan2(dz, dx))
    return min(angle, 90.0 - angle)


def test_an_l_walk_over_sparse_points_levels_scales_and_aligns_the_room():
    rng = np.random.default_rng(3)
    transform = colmap_transform()
    room, kind = make_room(rng)
    centres, forwards = l_walk(rng)
    images = [
        camera_in_colmap(i + 1, c, f, transform)
        for i, (c, f) in enumerate(zip(centres, forwards, strict=True))
    ]
    sparse = to_sparse(room, transform, rng)

    est = frame_from_sparse(sparse, images, seed=0)

    assert similarity_matrix_problem(est.matrix) is None
    assert est.method == "floor_plane"
    assert est.wall_yaw_source == "points"
    world = apply_affine(est.matrix, sparse.xyz)
    # The spec asks for 3 cm on the floor. With 1 cm noise the largest of 700 floor samples
    # reaches about 4 sigma (4.0 cm in this fixture), so the maximum is held to 4.5 cm, the
    # 95th percentile to 3 cm and the median to 1 cm.
    floor_y = np.abs(world[kind == FLOOR, 1])
    assert floor_y.max() < 0.045
    assert np.percentile(floor_y, 95) < 0.03
    assert np.median(floor_y) < 0.01
    assert est.meters_per_unit == pytest.approx(1 / COLMAP_SCALE, rel=0.03)
    cameras = apply_affine(est.matrix, np.array([im.center() for im in images]))
    assert np.median(cameras[:, 1]) == pytest.approx(CAMERA_HEIGHT_METERS, abs=0.03)
    for wall in range(2, 6):
        assert _axis_residual_degrees(world[kind == wall][:, [0, 2]]) < 1.5
    # The matrix itself is recentred: the camera path's XZ bounds, with the margin, centre on
    # the origin, and the reported bounds centre on the same point.
    lo = cameras[:, [0, 2]].min(axis=0) - PATH_MARGIN_METERS
    hi = cameras[:, [0, 2]].max(axis=0) + PATH_MARGIN_METERS
    assert (lo + hi) / 2 == pytest.approx([0.0, 0.0], abs=0.01)
    reported = (np.array(est.min_xz) + np.array(est.max_xz)) / 2
    assert reported == pytest.approx((lo + hi) / 2, abs=0.01)
    assert reported == pytest.approx([0.0, 0.0], abs=1e-9)
    # The spawn is where the walk started (the first camera by name), on the floor, facing
    # along that camera's heading.
    first = min(images, key=lambda image: image.name)
    start = apply_affine(est.matrix, np.array([first.center()]))[0]
    assert est.spawn_feet[0] == pytest.approx(start[0], abs=0.01)
    assert est.spawn_feet[1] == 0.0
    assert est.spawn_feet[2] == pytest.approx(start[2], abs=0.01)
    linear = np.array(est.matrix).reshape(4, 4).T[:3, :3]
    heading = linear @ first.forward()
    expected_yaw = math.atan2(-heading[0], -heading[2])
    yaw_error = (est.spawn_yaw - expected_yaw + math.pi) % (2 * math.pi) - math.pi
    assert yaw_error == pytest.approx(0.0, abs=1e-6)
    # The spawn faces where the first camera looked.
    ends = apply_affine(est.matrix, np.array([first.center(), first.center() + first.forward()]))
    look = ends[1] - ends[0]
    look = np.array([look[0], 0.0, look[2]]) / np.linalg.norm([look[0], look[2]])
    facing = np.array([-math.sin(est.spawn_yaw), 0.0, -math.cos(est.spawn_yaw)])
    assert facing @ look > 0.999
    summary = est.summary()
    assert summary["wallYawSource"] == "points"
    assert "wallYawDegrees" in summary


def test_a_straight_walk_without_yaw_variation_still_recovers_up():
    rng = np.random.default_rng(9)
    n = 1200
    floor = np.column_stack([rng.uniform(-1, 6, n), np.zeros(n), rng.uniform(-1, 1, n)])
    walls = np.column_stack(
        [rng.uniform(-1, 6, n), rng.uniform(0, 2.5, n), rng.choice([-1.0, 1.0], n)]
    )
    ceiling = np.column_stack([rng.uniform(-1, 6, n), np.full(n, 2.5), rng.uniform(-1, 1, n)])
    room = np.vstack([floor, walls, ceiling]) + rng.normal(0, 0.01, (3 * n, 3))
    centres = np.column_stack([np.linspace(0, 5, 30), np.full(30, 1.5), np.zeros(30)])
    pitch = math.radians(20.0)
    forwards = np.tile([math.cos(pitch), -math.sin(pitch), 0.0], (30, 1))
    transform = colmap_transform()
    images = [
        camera_in_colmap(i + 1, c, f, transform)
        for i, (c, f) in enumerate(zip(centres, forwards, strict=True))
    ]
    sparse = to_sparse(room, transform, rng)

    est = frame_from_sparse(sparse, images, seed=0)

    assert similarity_matrix_problem(est.matrix) is None
    colmap_up = transform[0] @ UP  # the canonical up direction, in COLMAP's world
    ends = apply_affine(est.matrix, np.array([[0.0, 0.0, 0.0], colmap_up]))
    world_up = ends[1] - ends[0]
    cosine = world_up @ UP / np.linalg.norm(world_up)
    assert math.degrees(math.acos(min(1.0, cosine))) < 2.0
    # The walls run along canonical X, and so does the walk: its steps give the wall
    # direction (mod 90 degrees) in the levelled frame.
    assert est.wall_yaw_source == "walk"
    level = rotation_to_z(transform[0] @ UP)
    wall = level @ (transform[0] @ np.array([1.0, 0.0, 0.0]))
    expected = math.degrees(math.atan2(wall[1], wall[0])) % 90.0
    assert (est.wall_yaw_degrees - expected + 45.0) % 90.0 - 45.0 == pytest.approx(0.0, abs=2.0)
    assert est.summary()["wallYawSource"] == "walk"


def test_sixty_sparse_points_give_a_frame_or_a_warning_not_a_crash():
    rng = np.random.default_rng(5)
    transform = colmap_transform()
    room, _ = make_room(rng)
    centres, forwards = l_walk(rng)
    images = cameras_for(centres, forwards, transform)
    # Sixty points drawn across the floor, the walls and the ceiling.
    chosen = rng.choice(len(room), 60, replace=False)
    est = frame_from_sparse(to_sparse(room[chosen], transform, rng), images, seed=0)
    assert similarity_matrix_problem(est.matrix) is None
    assert est.method in ("floor_plane", "trainer_up")
    # Below SPARSE_WALL_MIN_POINTS the points give no wall direction, whatever the cloud;
    # the L-shaped walk runs along the walls.
    assert est.wall_yaw_source == "walk"
    assert "wallYawSource" in est.summary()


def test_a_featureless_scene_with_turning_headings_has_no_wall_direction():
    rng = np.random.default_rng(13)
    transform = colmap_transform()
    # Unstructured points, and a camera that turns once in place through every heading:
    # the headings mod 90 degrees are flat, so neither source finds a wall.
    cloud = to_sparse(rng.uniform([0, 0, 0], [5, 2.6, 3.5], (300, 3)), transform, rng)
    headings = np.radians(np.arange(360.0))
    pitch = math.radians(15.0)
    centres = np.tile([2.5, 1.5, 1.75], (360, 1))
    forwards = np.column_stack([
        np.cos(headings) * math.cos(pitch),
        np.full(360, -math.sin(pitch)),
        np.sin(headings) * math.cos(pitch),
    ])  # fmt: skip
    est = frame_from_sparse(cloud, cameras_for(centres, forwards, transform), seed=0)
    assert similarity_matrix_problem(est.matrix) is None
    assert est.wall_yaw_source == "none"
    assert est.wall_yaw_degrees == 0.0
    assert "wallYawSource" not in est.summary()


def test_a_corridor_walk_aligns_the_walls_even_when_the_phone_looks_around():
    # The first real capture: walking down a corridor while the phone sweeps 45 degrees to
    # one side, then turning on the spot to face a side wall. The headings disagree with
    # the walls; the walk's steps do not.
    rng = np.random.default_rng(21)
    transform = colmap_transform()
    n = 24  # frames about a second apart: steps of a third of a metre, swaying a little
    along = np.concatenate([np.linspace(0, 5, n - 8), np.full(8, 5.0)])
    centres = np.column_stack(
        [along + rng.normal(0, 0.03, n), np.full(n, 1.5), rng.normal(0, 0.04, n)]
    )
    sweep = np.concatenate([np.linspace(-45, 0, n - 8), np.linspace(0, -90, 8)])
    pitch = math.radians(15.0)
    forwards = np.column_stack([
        np.cos(np.radians(sweep)) * math.cos(pitch),
        np.full(n, -math.sin(pitch)),
        np.sin(np.radians(sweep)) * math.cos(pitch),
    ])  # fmt: skip
    # Too few points for a wall direction of their own.
    cloud = to_sparse(rng.uniform([-1, 0, -0.6], [6, 2.5, 0.6], (150, 3)), transform, rng)

    est = frame_from_sparse(cloud, cameras_for(centres, forwards, transform), seed=0)

    assert est.wall_yaw_source == "walk"
    a, scale, b = transform
    world = apply_affine(est.matrix, scale * centres @ a.T + b)
    steps = np.diff(world[:, [0, 2]], axis=0)
    angles = np.degrees(np.arctan2(steps[:, 1], steps[:, 0]))
    moving = np.linalg.norm(steps, axis=1) > 0.05
    folded = (angles[moving] + 45.0) % 90.0 - 45.0
    assert abs(float(np.median(folded))) < 2.0  # the corridor runs along a world axis


def test_a_few_random_steps_do_not_claim_a_wall_direction():
    rng = np.random.default_rng(3)
    transform = colmap_transform()
    for seed in range(20):
        steps = np.random.default_rng(seed).uniform(-np.pi, np.pi, 6)
        offsets = np.column_stack([np.cos(steps), np.zeros(6), np.sin(steps)]) * 0.5
        centres = np.vstack([[0.0, 1.5, 0.0], np.cumsum(offsets, axis=0) + [0.0, 1.5, 0.0]])
        headings = np.radians(np.arange(0.0, 360.0, 360.0 / len(centres)))
        forwards = np.column_stack([np.cos(headings), np.zeros(len(centres)), np.sin(headings)])
        cloud = to_sparse(rng.uniform([-3, 0, -3], [3, 2.5, 3], (150, 3)), transform, rng)
        est = frame_from_sparse(cloud, cameras_for(centres, forwards, transform), seed=0)
        assert est.wall_yaw_source != "walk", seed


def test_untrusted_points_are_left_out_of_the_frame():
    rng = np.random.default_rng(11)
    transform = colmap_transform()
    room, _ = make_room(rng)
    centres, forwards = l_walk(rng)
    images = cameras_for(centres, forwards, transform)
    sparse = to_sparse(room, transform, rng)
    # 300 stray points inside the room, in three kinds: a 3 px error (fails the 2 px
    # limit), a track of one (fails the track test), and a 5 px error on a track of one.
    stray_xyz = rng.uniform([0, 0, 0], [5, 2.6, 3.5], (300, 3))
    stray = with_trust(
        to_sparse(stray_xyz, transform, rng),
        np.repeat([3.0, 0.5, 5.0], 100),
        np.repeat([5, 1, 1], 100),
    )
    clean = frame_from_sparse(sparse, images, seed=0)
    with_stray = frame_from_sparse(join_points(sparse, stray), images, seed=0)
    assert with_stray.matrix == pytest.approx(clean.matrix, abs=1e-9)
    assert with_stray.wall_yaw_source == clean.wall_yaw_source
    # The same points would move the frame if they were trusted.
    trusted = with_trust(stray, np.full(300, 0.5), np.full(300, 5))
    moved = frame_from_sparse(join_points(sparse, trusted), images, seed=0)
    assert np.abs(np.array(moved.matrix) - np.array(clean.matrix)).max() > 1e-2


def test_a_cloud_with_fewer_than_fifty_trusted_points_uses_the_four_pixel_limit():
    rng = np.random.default_rng(12)
    transform = colmap_transform()
    room, _ = make_room(rng)
    centres, forwards = l_walk(rng)
    images = cameras_for(centres, forwards, transform)
    sparse = to_sparse(room, transform, rng)
    # Only 30 points pass the 2 px limit; the other 1970 pass 4 px; 200 stray points fail it.
    error = np.full(len(room), 3.0)
    error[:30] = 0.5
    stray_xyz = rng.uniform([0, 0, 0], [5, 2.6, 3.5], (200, 3))
    stray = with_trust(to_sparse(stray_xyz, transform, rng), np.full(200, 5.0), np.full(200, 5))
    combined = join_points(with_trust(sparse, error, np.full(len(room), 5)), stray)
    assert len(reliable_points(combined)) == len(room)
    expected = frame_from_sparse(sparse, images, seed=0)
    got = frame_from_sparse(combined, images, seed=0)
    assert got.matrix == pytest.approx(expected.matrix, abs=1e-9)


def test_a_cloud_with_fewer_than_twenty_trusted_points_uses_every_point():
    rng = np.random.default_rng(14)
    transform = colmap_transform()
    room, _ = make_room(rng)
    centres, forwards = l_walk(rng)
    images = cameras_for(centres, forwards, transform)
    sparse = to_sparse(room, transform, rng)
    # Ten points pass the 2 px limit; the other 1990 fail even the 4 px limit, so every point
    # is used.
    error = np.full(len(room), 6.0)
    error[:10] = 0.5
    weak = with_trust(sparse, error, np.full(len(room), 5))
    assert len(reliable_points(weak)) == len(room)
    expected = frame_from_sparse(sparse, images, seed=0)
    assert frame_from_sparse(weak, images, seed=0).matrix == pytest.approx(
        expected.matrix, abs=1e-9
    )


def test_a_point_at_exactly_two_pixels_is_not_trusted():
    rng = np.random.default_rng(19)
    transform = colmap_transform()
    # Sixty clean points, thirty at 1.99 px (trusted) and one hundred at exactly 2.0 px (not).
    error = np.concatenate([np.full(60, 0.5), np.full(30, 1.99), np.full(100, 2.0)])
    cloud = with_trust(
        to_sparse(rng.uniform(0, 5, (190, 3)), transform, rng), error, np.full(190, 5)
    )
    np.testing.assert_array_equal(reliable_points(cloud), cloud.xyz[:90])


def test_frame_from_sparse_needs_a_camera():
    rng = np.random.default_rng(16)
    transform = colmap_transform()
    room, _ = make_room(rng)
    with pytest.raises(ValueError, match="no cameras"):
        frame_from_sparse(to_sparse(room, transform, rng), [], seed=0)


def test_frame_from_sparse_needs_a_point():
    rng = np.random.default_rng(16)
    transform = colmap_transform()
    centres, forwards = l_walk(rng)
    no_points = to_sparse(np.empty((0, 3)), transform, rng)
    with pytest.raises(ValueError, match="no sparse points"):
        frame_from_sparse(no_points, cameras_for(centres, forwards, transform), seed=0)


def test_the_frame_does_not_depend_on_the_order_the_images_are_given_in():
    rng = np.random.default_rng(15)
    transform = colmap_transform()
    room, _ = make_room(rng)
    centres, forwards = l_walk(rng)
    images = cameras_for(centres, forwards, transform)
    sparse = to_sparse(room, transform, rng)
    forward = frame_from_sparse(sparse, images, seed=0)
    backward = frame_from_sparse(sparse, images[::-1], seed=0)
    assert backward.matrix == pytest.approx(forward.matrix, abs=1e-9)
    assert backward.spawn_feet == pytest.approx(forward.spawn_feet, abs=1e-9)


def test_a_sparse_floor_is_found_with_the_fractional_inlier_minimum():
    rng = np.random.default_rng(6)
    floor = np.column_stack(
        [rng.uniform(0, 5, 150), rng.uniform(0, 3.5, 150), rng.normal(0, 0.005, 150)]
    )
    centres = np.column_stack([np.linspace(0.5, 4.5, 20), np.full(20, 1.75), np.full(20, 1.5)])
    path = CameraPath(centres, np.tile([1.0, 0.0, 0.0], (20, 1)))
    assert estimate_frame(floor, path, seed=0).method == "trainer_up"
    fractional = estimate_frame(floor, path, seed=0, min_inliers_fraction=0.05)
    assert fractional.method == "floor_plane"
    assert "wallYawSource" not in fractional.summary()


@pytest.mark.parametrize(("count", "method"), [(19, "trainer_up"), (20, "floor_plane")])
def test_a_fractional_inlier_minimum_never_drops_below_twenty(count, method):
    rng = np.random.default_rng(17)
    # Every point lies on the floor near the path, so 5 % of the cloud is at most one point
    # and the floor of twenty inliers is what decides.
    floor = np.column_stack([
        rng.uniform(0, 3, count), rng.uniform(0, 2, count), rng.normal(0, 0.005, count)
    ])  # fmt: skip
    centres = np.column_stack([np.linspace(0.5, 2.5, 20), np.full(20, 1.0), np.full(20, 1.5)])
    path = CameraPath(centres, np.tile([1.0, 0.0, 0.0], (20, 1)))
    est = estimate_frame(floor, path, seed=0, min_inliers_fraction=0.05)
    assert est.method == method


def _u_walk(rng: np.random.Generator, length: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    """A U-shaped walk along three sides of a room: 1.5 m up, pitched 15 degrees down.

    Each leg has 14 frames, and each heading wanders by 8 degrees.
    """
    pitch = math.radians(15.0)
    near, far_x, far_z = 0.6, length - 0.6, width - 0.6
    legs = [
        ((near, near), (far_x, near)),
        ((far_x, near), (far_x, far_z)),
        ((far_x, far_z), (near, far_z)),
    ]
    centres, forwards = [], []
    for (x0, z0), (x1, z1) in legs:
        base = math.atan2(z1 - z0, x1 - x0)
        for t in np.linspace(0.0, 1.0, 14, endpoint=False):
            centres.append([x0 + (x1 - x0) * t, 1.5, z0 + (z1 - z0) * t])
            yaw = base + rng.normal(0.0, math.radians(8.0))
            heading = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
            forwards.append(heading * math.cos(pitch) - UP * math.sin(pitch))
    return np.array(centres), np.array(forwards)


def test_a_u_shaped_walk_in_a_rectangular_room_aligns_its_walls_too():
    # Not an L: three legs, a 6 x 4 m room, and walls along X and Z that the walk does not
    # follow on a single leg.
    rng = np.random.default_rng(21)
    transform = colmap_transform()
    room, kind = make_room(rng, length=6.0, width=4.0, height=2.5)
    centres, forwards = _u_walk(rng, 6.0, 4.0)
    images = cameras_for(centres, forwards, transform)
    sparse = to_sparse(room, transform, rng)

    est = frame_from_sparse(sparse, images, seed=0)

    assert similarity_matrix_problem(est.matrix) is None
    assert est.method == "floor_plane"
    assert est.wall_yaw_source == "points"
    world = apply_affine(est.matrix, sparse.xyz)
    floor_y = np.abs(world[kind == FLOOR, 1])
    assert np.percentile(floor_y, 95) < 0.03
    assert np.median(floor_y) < 0.01
    assert est.meters_per_unit == pytest.approx(1 / COLMAP_SCALE, rel=0.03)
    cameras = apply_affine(est.matrix, np.array([im.center() for im in images]))
    assert np.median(cameras[:, 1]) == pytest.approx(CAMERA_HEIGHT_METERS, abs=0.03)
    for wall in range(2, 6):
        assert _axis_residual_degrees(world[kind == wall][:, [0, 2]]) < 1.5


@pytest.mark.parametrize("jitter", [0.0, 1e-3])
def test_a_camera_that_barely_moves_takes_the_mean_image_up(jitter):
    rng = np.random.default_rng(8)
    # Right vectors that never change, so the spread test sends the search to the walk
    # branch; the camera hardly moves (1 mm of jitter against a 5 m extent), so the mean
    # image-up is the answer.
    centres = np.array([1.0, 1.5, 1.0]) + rng.normal(0.0, jitter, (6, 3))
    rights = np.tile([1.0, 0.0, 0.0], (6, 1))
    ups = np.tile([0.05, 0.98, 0.1], (6, 1))
    forwards = np.tile([0.0, 0.0, 1.0], (6, 1))
    up = _up_from_cameras(centres, rights, ups, forwards, 5.0)
    mean = ups.mean(axis=0)
    np.testing.assert_allclose(up, mean / np.linalg.norm(mean), atol=1e-9)


def test_a_straight_walk_takes_up_as_right_crossed_with_the_walk():
    # Right is +Z and the walk is +X, so right x walk = +Y: the moving branch. The phone is
    # pitched 25 degrees down, so each image-up tilts 25 degrees forward: their mean is 25
    # degrees off the true up, and only the cross product (right stays level) is within 1.
    pitch = math.radians(25.0)
    centres = np.column_stack([np.linspace(0, 5, 10), np.full(10, 1.5), np.zeros(10)])
    rights = np.tile([0.0, 0.0, 1.0], (10, 1))
    ups = np.tile([math.sin(pitch), math.cos(pitch), 0.0], (10, 1))
    forwards = np.tile([math.cos(pitch), -math.sin(pitch), 0.0], (10, 1))
    true_up = np.array([0.0, 1.0, 0.0])
    mean = ups.mean(axis=0) / np.linalg.norm(ups.mean(axis=0))
    assert np.degrees(np.arccos(mean @ true_up)) == pytest.approx(25.0, abs=1e-6)
    up = _up_from_cameras(centres, rights, ups, forwards, 5.0)
    assert np.degrees(np.arccos(np.clip(up @ true_up, -1.0, 1.0))) < 1.0


@pytest.mark.parametrize("n", [3, 40, 700, 1500])
def test_nearest_neighbours_match_a_brute_force_search(n):
    rng = np.random.default_rng(n)
    # A solid block and a thin sheet, so the grid cells hold very different counts.
    points = np.vstack([
        rng.uniform(0, 4, (n // 2, 3)),
        rng.uniform(0, 4, (n - n // 2, 3)) * [1.0, 1.0, 0.01],
    ])  # fmt: skip
    k = min(6, n - 1)
    indices, dist = _nearest_neighbours(points, np.arange(n), k)
    full = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(full, np.inf)
    np.testing.assert_allclose(np.sort(dist, axis=1), np.sort(full, axis=1)[:, :k], atol=1e-12)
    assert not np.any(indices == np.arange(n)[:, None])  # a point is not its own neighbour
    np.testing.assert_allclose(
        np.linalg.norm(points[indices] - points[:, None, :], axis=2), dist, atol=1e-12
    )


def test_nearest_neighbours_of_identical_points_are_all_at_zero_distance():
    indices, dist = _nearest_neighbours(np.zeros((10, 3)), np.arange(10), 6)
    assert np.all(dist == 0.0)
    assert not np.any(indices == np.arange(10)[:, None])
