"""Up, floor, estimated scale and spawn from the camera path (geometry.frame)."""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from andarilho_worker.geometry.frame import (
    CAMERA_HEIGHT_METERS,
    CameraPath,
    colmap_to_export,
    estimate_frame,
    rotation_to_z,
)
from andarilho_worker.stages.reconstruct import read_camera_poses
from andarilho_worker.validation import similarity_matrix_problem


def _apply(matrix: list[float], points: np.ndarray) -> np.ndarray:
    m = np.array(matrix).reshape(4, 4).T
    return points @ m[:3, :3].T + m[:3, 3]


def _axis_rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = math.radians(degrees)
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k


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
    tilt_r = _axis_rotation(np.array([0.3, 1.0, 0.0]), tilt)
    scale, offset = 0.21, np.array([0.4, -0.2, 0.3])
    export = points @ tilt_r.T * scale + offset
    path = CameraPath(centers @ tilt_r.T * scale + offset, forwards @ tilt_r.T)

    est = estimate_frame(export, path, seed=3)

    assert est.method == "floor_plane"
    assert similarity_matrix_problem(est.matrix) is None
    assert est.tilt_degrees == pytest.approx(tilt, abs=1.0)
    assert est.meters_per_unit == pytest.approx(1 / scale, rel=0.03)
    world_floor = _apply(est.matrix, points[:6000] @ tilt_r.T * scale + offset)
    assert np.abs(world_floor[:, 1]).max() < 0.05  # the floor is level at y = 0
    world_cams = _apply(est.matrix, path.centers)
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


def test_colmap_to_export_follows_nerfstudio_world_swap_transform_and_scale():
    transform = [[0, 1, 0, 0.5], [1, 0, 0, 0], [0, 0, -1, 1]]
    p = np.array([[1.0, 2.0, 3.0]])  # COLMAP world
    # Nerfstudio: (x, y, z) -> (x, z, -y) = (1, 3, -2); transform -> (3, 1, 3) + t.
    expected = (np.array([3.0 + 0.5, 1.0, 2.0 + 1.0]) * 0.5) - np.array([0.1, 0.2, 0.3])
    out = colmap_to_export(p, transform, 0.5, [0.1, 0.2, 0.3])
    assert out[0] == pytest.approx(expected)
    d = colmap_to_export(p, transform, 0.5, [0.1, 0.2, 0.3], directions=True)
    assert np.linalg.norm(d[0]) == pytest.approx(1.0)


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
