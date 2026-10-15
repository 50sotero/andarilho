"""Synthetic captures for the scene-analysis tests: posed cameras looking at a known room.

Every 2D detection is made by projecting true 3D boxes through a camera, so each test knows
the answer the grounding has to recover. The camera model matches what the analysis stores:
a pinhole with ``intrinsics`` [fx, fy, cx, cy] and a world-to-camera pose ``x_cam = R X + t``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

WIDTH, HEIGHT = 1000, 750
FOCAL = 650.0  # a wide lens, so objects a few metres away keep their bottom edge in view
INTRINSICS = [FOCAL, FOCAL, WIDTH / 2, HEIGHT / 2]
EYE = 1.5  # metres, the camera height the normalizing stage assumes


def look_at(centre: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    """The world-to-camera (R, t) of a camera at ``centre`` looking at ``target``.

    OpenCV axes: x right, y down, z forward, with the world's +Y as up.
    """
    centre = np.asarray(centre, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - centre
    forward /= np.linalg.norm(forward)
    down = -(np.array([0.0, 1.0, 0.0]) - forward[1] * forward)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    rotation = np.vstack([right, down, forward])
    return rotation, -rotation @ centre


def keyframe(frame: int, centre: Any, target: Any, *, name: str | None = None) -> dict[str, Any]:
    """A keyframe record of the analysis artifact, looking from ``centre`` at ``target``."""
    rotation, translation = look_at(centre, target)
    return {
        "frame": frame,
        "name": name or f"frame-{frame:04d}.jpg",
        "kind": "video",
        "width": WIDTH,
        "height": HEIGHT,
        "intrinsics": INTRINSICS,
        "rotation": rotation.reshape(-1).tolist(),
        "translation": translation.tolist(),
    }


def camera_centre(record: dict[str, Any]) -> np.ndarray:
    rotation = np.asarray(record["rotation"], dtype=np.float64).reshape(3, 3)
    return -rotation.T @ np.asarray(record["translation"], dtype=np.float64)


def box_corners(
    centre_xz: tuple[float, float],
    yaw: float,
    size: tuple[float, float, float],
    base: float = 0.0,
) -> np.ndarray:
    """The 8 corners of a box on the floor: footprint centre, yaw and (width, height, depth).

    Uses the object frame of the layout contract: local x across the width, z front to back,
    and the world point as ``x = px + lx cos + lz sin``, ``z = pz - lx sin + lz cos``.
    """
    width, height, depth = size
    px, pz = centre_xz
    corners = []
    for lx in (-width / 2, width / 2):
        for lz in (-depth / 2, depth / 2):
            for y in (base, base + height):
                x = px + lx * math.cos(yaw) + lz * math.sin(yaw)
                z = pz - lx * math.sin(yaw) + lz * math.cos(yaw)
                corners.append((x, y, z))
    return np.array(corners, dtype=np.float64)


def project(points: np.ndarray, record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Pixel fractions (x, y, down) of world points, and their depths in the camera."""
    rotation = np.asarray(record["rotation"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(record["translation"], dtype=np.float64)
    fx, fy, cx, cy = record["intrinsics"]
    cam = points @ rotation.T + translation
    depth = cam[:, 2]
    u = fx * cam[:, 0] / depth + cx
    v = fy * cam[:, 1] / depth + cy
    return np.column_stack([u / WIDTH, v / HEIGHT]), depth


def detection(record: dict[str, Any], corners: np.ndarray) -> dict[str, Any] | None:
    """The 2D box an analysis would report for a 3D box, or None when it is behind the camera.

    The box is the bounding box of the projected corners, clipped to the image. The bottom
    is visible when the bottom corners are inside the image; a cut-off bottom reads as 1.
    """
    fractions, depth = project(corners, record)
    if depth.min() <= 0.05:
        return None
    x0, x1 = float(fractions[:, 0].min()), float(fractions[:, 0].max())
    y0, y1 = float(fractions[:, 1].min()), float(fractions[:, 1].max())
    bottom = corners[:, 1] < 1e-9
    bottom_visible = bool(np.all((fractions[bottom] >= 0.0) & (fractions[bottom] <= 1.0)))
    box = [
        round(min(max(x0, 0.0), 1.0), 4),
        round(min(max(y0, 0.0), 1.0), 4),
        round(min(max(x1, 0.0), 1.0), 4),
        round(min(max(y1, 0.0), 1.0), 4),
    ]
    return {"box": box, "bottomVisible": bottom_visible}


def door_detection(
    record: dict[str, Any],
    centre_xz: tuple[float, float],
    yaw: float,
    width: float,
    height: float,
    *,
    state: str = "closed",
    leads_to: str | None = None,
) -> dict[str, Any] | None:
    """A door box in a wall. Its footprint is a thin slab at the wall, ``height`` tall."""
    slab = box_corners(centre_xz, yaw, (width, height, 0.02))
    found = detection(record, slab)
    if found is None:
        return None
    return {**found, "state": state, "leadsTo": leads_to}


def analysis_artifact(
    keyframes: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    *,
    rooms: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "status": "complete",
        "reason": None,
        "model": "test-model",
        "floorY": 0.0,
        "keyframes": keyframes,
        "reply": {"rooms": rooms or [], "frames": frames, "objects": objects},
    }


def agent_object(
    object_id: str,
    category: str,
    size: tuple[float, float, float],
    *,
    mount: str = "floor",
    solid: bool = True,
    label: str | None = None,
) -> dict[str, Any]:
    """An agent object with one box part, the way the analysis draws it."""
    return {
        "id": object_id,
        "category": category,
        "label": label or f"a {category}",
        "mount": mount,
        "solid": solid,
        "size": list(size),
        "parts": [
            {"shape": "box", "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0], "color": "#8a6d3b"}
        ],
    }
