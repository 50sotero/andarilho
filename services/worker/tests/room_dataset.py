"""A box room's capture dataset: frames rendered by ray casting, posed in a COLMAP frame.

Shared by the texturing stage and smoke tests. The archive has the layout of the one
``camera_reconstruction`` stores: ``images/`` plus ``sparse/0/`` with PINHOLE cameras.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from andarilho_worker.geometry.colmap import (
    ColmapCamera,
    ColmapImage,
    SparsePoints,
    write_cameras_bin,
    write_images_bin,
    write_points3d_bin,
)

WIDTH, HEIGHT = 160, 120
K = np.array([[110.0, 0.0, 80.0], [0.0, 110.0, 60.0], [0.0, 0.0, 1.0]])
# The room in the estimated (uncalibrated) frame, in metres: x, y (up), z.
ROOM_LO = np.array([-1.5, 0.0, -1.25])
ROOM_HI = np.array([1.5, 2.6, 1.25])
# One colour per face: x low (west), x high (east), floor, ceiling, z low, z high.
FACE_COLOURS = np.array(
    [(200, 70, 60), (60, 160, 90), (70, 110, 210), (210, 160, 60), (150, 70, 190), (60, 180, 190)],
    dtype=np.float64,
)
SCALE = 2.0  # the similarity's scale: COLMAP units to metres
YAW = 0.3
TRANSLATION = np.array([0.5, -0.2, 1.0])
# Vertical view is +-28.6 degrees (the frame is 160 x 120 at focal length 110 px). Steep
# pitches reach the floor and ceiling close to the camera; shallow ones reach the walls' ends.
PITCHES = (-0.9, -0.35, 0.35, 0.9)
CAMERAS = 24


def world_from_colmap() -> np.ndarray:
    """The 4x4 similarity from COLMAP to the canonical world."""
    c, s = np.cos(YAW), np.sin(YAW)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    matrix = np.eye(4)
    matrix[:3, :3] = SCALE * rotation
    matrix[:3, 3] = TRANSLATION
    return matrix


def quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """(w, x, y, z) of a rotation matrix (Shepperd's method, the largest term first)."""
    trace = np.trace(R)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        return (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    if R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)


def canonical_cameras() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(R, t, eye) of each camera in the canonical frame: inside the room, sweeping it."""
    poses = []
    up = np.array([0.0, 1.0, 0.0])
    for k in range(CAMERAS):
        yaw = 2.0 * np.pi * k / CAMERAS
        pitch = PITCHES[k % len(PITCHES)]
        eye = np.array([0.15 * np.cos(yaw), 1.4, 0.15 * np.sin(yaw)])
        forward = np.array(
            [np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)]
        )
        x_axis = np.cross(forward, up)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(forward, x_axis)  # down in the image, as COLMAP and OpenCV have it
        R = np.stack([x_axis, y_axis, forward])
        poses.append((R, -R @ eye, eye))
    return poses


def render_room(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """RGB of the box room seen from a canonical world-to-camera pose, by ray casting."""
    u, v = np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5)
    rays = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3) @ np.linalg.inv(K).T
    dirs = rays @ R  # camera rays in the world, as row vectors
    centre = -R.T @ t
    best = np.full(len(dirs), np.inf)
    face = np.zeros(len(dirs), dtype=np.int64)
    for axis in range(3):
        for side, plane in enumerate((ROOM_LO[axis], ROOM_HI[axis])):
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (plane - centre[axis]) / dirs[:, axis]
            hit = centre + s[:, None] * dirs
            ok = np.isfinite(s) & (s > 1e-9)
            for other in range(3):
                if other != axis:
                    ok &= (hit[:, other] >= ROOM_LO[other] - 1e-9) & (
                        hit[:, other] <= ROOM_HI[other] + 1e-9
                    )
            closer = ok & (s < best)
            best[closer] = s[closer]
            face[closer] = axis * 2 + side
    colours = FACE_COLOURS[np.where(np.isfinite(best), face, 0)]
    return np.rint(colours).reshape(HEIGHT, WIDTH, 3).astype(np.uint8)


def write_room_dataset(work: Path, tar_path: Path) -> np.ndarray:
    """Write the room's ``dataset.tar`` to ``tar_path``, using ``work`` as scratch.

    Returns the 4x4 similarity from the dataset's COLMAP frame to the canonical world.
    """
    (work / "images" / "00").mkdir(parents=True)
    (work / "sparse" / "0").mkdir(parents=True)
    world = world_from_colmap()
    # Canonical X = s R_w X_c + t_w, so a canonical pose (R, t) is the COLMAP pose
    # (R R_w, (R t_w + t) / s): the same pixels, seen from the COLMAP frame.
    R_w, t_w = world[:3, :3] / SCALE, world[:3, 3]
    images = []
    for k, (R, t_canonical, _eye) in enumerate(canonical_cameras()):
        name = f"00/frame_{k + 1:05d}.jpg"
        Image.fromarray(render_room(R, t_canonical)).save(
            work / "images" / name, "JPEG", quality=95
        )
        R_colmap, t_colmap = R @ R_w, (R @ t_w + t_canonical) / SCALE
        images.append(ColmapImage(k + 1, 1, name, quaternion(R_colmap), tuple(t_colmap)))
    write_cameras_bin(
        work / "sparse" / "0" / "cameras.bin",
        [ColmapCamera(1, "PINHOLE", WIDTH, HEIGHT, (110.0, 110.0, 80.0, 60.0))],
    )
    write_images_bin(work / "sparse" / "0" / "images.bin", images)
    points = SparsePoints(
        xyz=np.array([[0.0, 1.0, 0.0], [1.0, 0.5, 0.5]]),
        rgb=np.zeros((2, 3), dtype=np.uint8),
        error=np.zeros(2),
        track_length=np.full(2, 3, dtype=np.int64),
    )
    write_points3d_bin(work / "sparse" / "0" / "points3D.bin", points)
    with tarfile.open(tar_path, "w") as tar:
        tar.add(work / "images", arcname="images")
        tar.add(work / "sparse", arcname="sparse")
    return world
