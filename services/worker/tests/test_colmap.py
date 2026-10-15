"""COLMAP binary readers, writers and pose helpers (geometry.colmap)."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pytest

from andarilho_worker.geometry.colmap import (
    ColmapCamera,
    ColmapImage,
    SparsePoints,
    read_camera_poses,
    read_cameras_bin,
    read_images_bin,
    read_points3d,
    read_points3d_full,
    write_cameras_bin,
    write_images_bin,
    write_points3d_bin,
)
from andarilho_worker.stages import reconstruct

HALF = math.sqrt(0.5)


def _cameras() -> list[ColmapCamera]:
    return [
        ColmapCamera(1, "PINHOLE", 640, 480, (500.0, 501.0, 320.0, 240.0)),
        ColmapCamera(7, "SIMPLE_PINHOLE", 1280, 720, (900.0, 640.0, 360.0)),
        ColmapCamera(9, "FULL_OPENCV", 4032, 3024, tuple(0.01 * i for i in range(12))),
    ]


def _images() -> list[ColmapImage]:
    return [
        ColmapImage(2, 1, "00/frame_00002.jpg", (HALF, 0.0, HALF, 0.0), (1.0, 2.0, 3.0)),
        ColmapImage(1, 1, "00/frame_00001.jpg", (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -4.0)),
    ]


def _observations() -> list[list[tuple[float, float, int]]]:
    """2D observations for each image of ``_images()``: none, then two.

    The last image in the file carries them, so a cut inside its observation block has no
    later record to fail on and must be caught by the block's own bounds check.
    """
    return [[], [(10.0, 20.0, 7), (1.5, 2.5, -1)]]


def _points() -> SparsePoints:
    return SparsePoints(
        xyz=np.array([[0.5, -1.0, 2.0], [3.0, 0.0, 1.5], [-2.0, 4.0, 0.25]]),
        rgb=np.array([[255, 0, 1], [10, 20, 30], [0, 0, 0]], dtype=np.uint8),
        error=np.array([0.5, 1.25, 3.0]),
        track_length=np.array([3, 0, 7], dtype=np.int64),
    )


def test_cameras_round_trip(tmp_path: Path):
    path = tmp_path / "cameras.bin"
    write_cameras_bin(path, _cameras())
    read = read_cameras_bin(path)
    assert read == {camera.id: camera for camera in _cameras()}


def test_images_round_trip_skips_observations_and_sorts_by_name(tmp_path: Path):
    path = tmp_path / "images.bin"
    write_images_bin(path, _images(), _observations())
    assert read_images_bin(path) == [_images()[1], _images()[0]]


def test_points_round_trip_keeps_colour_error_and_track_length(tmp_path: Path):
    path = tmp_path / "points3D.bin"
    points = _points()
    write_points3d_bin(path, points)
    read = read_points3d_full(path)
    assert np.array_equal(read.xyz, points.xyz)
    assert np.array_equal(read.rgb, points.rgb)
    assert np.array_equal(read.error, points.error)
    assert np.array_equal(read.track_length, points.track_length)
    assert read_points3d(path) == pytest.approx(points.xyz)


def test_an_empty_points_file_reads_as_no_points(tmp_path: Path):
    path = tmp_path / "points3D.bin"
    write_points3d_bin(path, SparsePoints(np.zeros((0, 3)), np.zeros((0, 3), np.uint8),
                                          np.zeros(0), np.zeros(0, np.int64)))  # fmt: skip
    read = read_points3d_full(path)
    assert read.xyz.shape == (0, 3)
    assert read.rgb.shape == (0, 3)
    assert read.track_length.shape == (0,)
    assert read_points3d(path).shape == (0, 3)


@pytest.mark.parametrize("kind", ["cameras", "images", "points3D", "points3D xyz only"])
def test_every_truncated_prefix_is_refused_with_a_value_error(tmp_path: Path, kind: str):
    path = tmp_path / "model.bin"
    writer, reader = {
        "cameras": (lambda p: write_cameras_bin(p, _cameras()), read_cameras_bin),
        # With observations, so every cut inside the 2D point block is covered too.
        "images": (lambda p: write_images_bin(p, _images(), _observations()), read_images_bin),
        "points3D": (lambda p: write_points3d_bin(p, _points()), read_points3d_full),
        "points3D xyz only": (lambda p: write_points3d_bin(p, _points()), read_points3d),
    }[kind]
    writer(path)
    data = path.read_bytes()
    for cut in range(len(data)):
        path.write_bytes(data[:cut])
        with pytest.raises(ValueError, match="truncated"):
            reader(path)


def test_a_name_that_is_not_utf8_is_refused(tmp_path: Path):
    blob = struct.pack("<Q", 1)
    blob += struct.pack("<idddddddi", 1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1)
    blob += b"\xff\xfe\x00" + struct.pack("<Q", 0)
    path = tmp_path / "images.bin"
    path.write_bytes(blob)
    with pytest.raises(ValueError, match="UTF-8"):
        read_images_bin(path)


def test_an_unknown_camera_model_is_refused(tmp_path: Path):
    path = tmp_path / "cameras.bin"
    path.write_bytes(struct.pack("<QiiQQ", 1, 1, 99, 640, 480) + bytes(8 * 4))
    with pytest.raises(ValueError, match="unknown COLMAP camera model 99"):
        read_cameras_bin(path)


def test_more_points_than_the_limit_are_refused_but_the_legacy_reader_truncates(tmp_path: Path):
    path = tmp_path / "points3D.bin"
    write_points3d_bin(path, _points())
    with pytest.raises(ValueError, match="above the limit"):
        read_points3d_full(path, limit=2)
    assert read_points3d(path, limit=2) == pytest.approx(_points().xyz[:2])


def test_intrinsics_for_pinhole_and_simple_pinhole_only():
    pinhole = ColmapCamera(1, "PINHOLE", 640, 480, (500.0, 501.0, 320.0, 240.0))
    assert pinhole.intrinsics() == pytest.approx(
        np.array([[500.0, 0.0, 320.0], [0.0, 501.0, 240.0], [0.0, 0.0, 1.0]])
    )
    simple = ColmapCamera(2, "SIMPLE_PINHOLE", 640, 480, (400.0, 310.0, 230.0))
    assert simple.intrinsics() == pytest.approx(
        np.array([[400.0, 0.0, 310.0], [0.0, 400.0, 230.0], [0.0, 0.0, 1.0]])
    )
    opencv = ColmapCamera(3, "OPENCV", 640, 480, tuple(0.0 for _ in range(8)))
    with pytest.raises(ValueError, match="PINHOLE"):
        opencv.intrinsics()


def test_pose_helpers_match_hand_computed_values_for_a_known_rotation():
    # Camera 2 of the reader tests: R = rot_y(90) = [[0,0,1],[0,1,0],[-1,0,0]], t = (1,2,3).
    image = _images()[0]
    assert image.rotation() == pytest.approx(np.array([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]]),
                                             abs=1e-12)  # fmt: skip
    # centre = -R^T t = (3, -2, -1); forward = R^T e_z = (-1, 0, 0);
    # right = R^T e_x = (0, 0, 1); up = R^T (0, -1, 0) = (0, -1, 0).
    assert image.center() == pytest.approx([3.0, -2.0, -1.0], abs=1e-12)
    assert image.forward() == pytest.approx([-1.0, 0.0, 0.0], abs=1e-12)
    assert image.right() == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)
    assert image.up() == pytest.approx([0.0, -1.0, 0.0], abs=1e-12)


def test_an_unnormalised_quaternion_gives_the_same_rotation():
    scaled = ColmapImage(3, 1, "x.jpg", (2 * HALF, 0.0, 2 * HALF, 0.0), (1.0, 2.0, 3.0))
    assert scaled.rotation() == pytest.approx(_images()[0].rotation(), abs=1e-12)


def test_a_zero_quaternion_is_refused():
    image = ColmapImage(3, 1, "x.jpg", (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="quaternion"):
        image.rotation()


def test_the_camera_pose_reader_is_the_moved_one(tmp_path: Path):
    path = tmp_path / "images.bin"
    write_images_bin(path, _images())
    assert reconstruct.read_camera_poses is read_camera_poses
    assert reconstruct.read_points3d is read_points3d
    poses = read_camera_poses(path)
    assert [pose["name"] for pose in poses] == ["00/frame_00001.jpg", "00/frame_00002.jpg"]
    assert poses[1]["center"] == pytest.approx([3.0, -2.0, -1.0])
