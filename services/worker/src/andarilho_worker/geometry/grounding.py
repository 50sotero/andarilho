"""Ground the scene analysis's 2D detections on the floor of the posed keyframes.

The analysis reports boxes in keyframes whose poses are known in the canonical world
(metres, +Y up, floor at y = 0). Where a box's bottom edge is in the image, its ray meets
the floor, so a door or a piece of furniture can be placed in metres from a single frame.
Furniture seen in several frames is fused into one estimate. Pure numpy, no storage.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

FLOOR_RANGE = 12.0  # metres: a floor hit farther out comes from a ray too near the horizon
BOTTOM_EDGE_LIMIT = 0.985  # a bottom edge this low in the image is cut off by the frame
TOP_EDGE_LIMIT = 0.015  # a top edge this high in the image is cut off by the frame
DEFAULT_DOOR_TOP = 2.05  # metres, when the top of the door is out of frame
OBJECT_MIN_DISTANCE = 0.3  # metres from the camera to a floor point that is used
OBJECT_MAX_DISTANCE = 10.0
TOP_EDGE_SLACK = 0.05  # metres around the camera height where a top edge is in front
MAX_OBJECT_CONFIDENCE = 0.9


@dataclass(frozen=True, eq=False)
class PosedView:
    """A keyframe's pixel model and its pose in the canonical world.

    ``R`` and ``t`` map a canonical point into the camera (``x_cam = R @ X + t``), as
    ``geometry.texture.camera_in_world`` returns them, so the camera centre is ``-R.T @ t``.
    """

    frame: int
    K: np.ndarray  # (3, 3) pixel intrinsics
    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)
    width: int
    height: int

    def centre(self) -> np.ndarray:
        return -self.R.T @ self.t

    def ray(self, fx: float, fy: float) -> np.ndarray:
        """The unit world direction through the image point at fractions (fx, fy)."""
        u, v = fx * self.width, fy * self.height
        local = np.array(
            [(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], 1.0]
        )
        direction = self.R.T @ local
        return direction / np.linalg.norm(direction)


def views_from_analysis(analysis: Mapping[str, Any]) -> dict[int, PosedView]:
    """The posed view of every keyframe, by frame number."""
    views: dict[int, PosedView] = {}
    for keyframe in analysis["keyframes"]:
        fx, fy, cx, cy = (float(v) for v in keyframe["intrinsics"])
        frame = int(keyframe["frame"])
        views[frame] = PosedView(
            frame=frame,
            K=np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
            R=np.asarray(keyframe["rotation"], dtype=np.float64).reshape(3, 3),
            t=np.asarray(keyframe["translation"], dtype=np.float64).reshape(3),
            width=int(keyframe["width"]),
            height=int(keyframe["height"]),
        )
    return views


def floor_hit(
    view: PosedView, fx: float, fy: float, floor_y: float = 0.0, max_range: float = FLOOR_RANGE
) -> np.ndarray | None:
    """Where the ray through image point (fx, fy) meets the floor, as (x, z).

    None when the ray does not go down, or when it meets the floor farther than
    ``max_range`` metres away horizontally.
    """
    direction = view.ray(fx, fy)
    if direction[1] >= 0.0:
        return None
    centre = view.centre()
    along = (floor_y - centre[1]) / direction[1]
    if along <= 0.0:
        return None
    point = centre + along * direction
    if math.hypot(point[0] - centre[0], point[2] - centre[2]) > max_range:
        return None
    return np.array([point[0], point[2]])


def _height_at(view: PosedView, direction: np.ndarray, distance: float) -> float | None:
    """The height of a ray from the camera at a horizontal distance, or None if vertical."""
    horizontal = math.hypot(direction[0], direction[2])
    if horizontal < 1e-9:
        return None
    return float(view.centre()[1] + distance / horizontal * direction[1])


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


# --------------------------------------------------------------------------
# Doors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundedDoor:
    frame: int
    position: tuple[float, float]  # midpoint of the bottom edge on the floor, (x, z)
    direction: tuple[float, float]  # unit, from the left jamb to the right jamb
    width: float  # metres
    top: float  # height of the top edge, metres
    state: str  # "open" or "closed", as the agent saw it
    leads_to: str | None


def _ground_door(view: PosedView, door: Mapping[str, Any]) -> GroundedDoor | None:
    x0, y0, x1, y1 = (float(v) for v in door["box"])
    if y1 >= BOTTOM_EDGE_LIMIT:
        return None
    left = floor_hit(view, x0, y1)
    right = floor_hit(view, x1, y1)
    if left is None or right is None:
        return None
    span = right - left
    width = float(np.hypot(span[0], span[1]))
    if width <= 1e-6:
        return None
    middle = (left + right) / 2
    top = DEFAULT_DOOR_TOP
    if y0 > TOP_EDGE_LIMIT:
        centre = view.centre()
        distance = float(np.hypot(middle[0] - centre[0], middle[1] - centre[2]))
        height = _height_at(view, view.ray((x0 + x1) / 2, y0), distance)
        if height is not None:
            top = height
    return GroundedDoor(
        frame=view.frame,
        position=(float(middle[0]), float(middle[1])),
        direction=(float(span[0] / width), float(span[1] / width)),
        width=width,
        top=top,
        state=str(door["state"]),
        leads_to=door.get("leadsTo"),
    )


def ground_doors(analysis: Mapping[str, Any], views: Mapping[int, PosedView]) -> list[GroundedDoor]:
    """Every door whose two bottom corners meet the floor in view. Duplicates are kept."""
    doors: list[GroundedDoor] = []
    for entry in analysis["reply"]["frames"]:
        view = views.get(int(entry["frame"]))
        if view is None:
            continue
        for door in entry.get("doors", []):
            grounded = _ground_door(view, door)
            if grounded is not None:
                doors.append(grounded)
    return doors


# --------------------------------------------------------------------------
# Furniture
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class GroundedObject:
    id: str
    category: str
    label: str
    solid: bool
    parts: list[dict[str, Any]]
    position: tuple[float, float]  # footprint centre on the floor, (x, z)
    size: tuple[float, float, float]  # width, height, depth in metres
    confidence: float
    detections: int


@dataclass(frozen=True)
class _Sighting:
    """One frame that shows a floor object's front-bottom edge on the floor."""

    floor: np.ndarray  # (x, z) where the bottom-centre ray meets the floor
    away: np.ndarray  # unit horizontal direction from the camera to ``floor``
    distance: float  # horizontal distance from the camera to ``floor``
    camera_height: float
    width: float | None  # metres between the bottom corners, when both meet the floor
    top_height: float | None  # height of the top-centre ray at ``distance``


def floor_objects(analysis: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The agent's objects that stand on the floor, by id (wall and ceiling items are not)."""
    return {o["id"]: o for o in analysis["reply"]["objects"] if o["mount"] == "floor"}


def _sight(view: PosedView, detection: Mapping[str, Any]) -> _Sighting | None:
    x0, y0, x1, y1 = (float(v) for v in detection["box"])
    if not detection["bottomVisible"] or y1 >= BOTTOM_EDGE_LIMIT:
        return None
    floor = floor_hit(view, (x0 + x1) / 2, y1)
    if floor is None:
        return None
    centre = view.centre()
    offset = floor - centre[[0, 2]]
    distance = float(np.hypot(offset[0], offset[1]))
    if not OBJECT_MIN_DISTANCE <= distance <= OBJECT_MAX_DISTANCE:
        return None
    left, right = floor_hit(view, x0, y1), floor_hit(view, x1, y1)
    width = None
    if left is not None and right is not None:
        width = float(np.hypot(*(right - left)))
    top_height = None
    if y0 > TOP_EDGE_LIMIT:
        top_height = _height_at(view, view.ray((x0 + x1) / 2, y0), distance)
    return _Sighting(
        floor=floor,
        away=offset / distance,
        distance=distance,
        camera_height=float(centre[1]),
        width=width,
        top_height=top_height,
    )


def _top_edge(sighting: _Sighting, top_height: float, depth: float) -> float:
    """The object's top height from one sighting, given its fused depth.

    A top edge at or above the camera is the front one, which the top-centre ray meets at
    the bottom's distance. A lower top edge is the back one, which the same ray meets
    ``depth`` metres farther out, so its height is read there.
    """
    camera_height = sighting.camera_height
    if top_height >= camera_height - TOP_EDGE_SLACK:
        return top_height
    ratio = (sighting.distance + depth) / sighting.distance
    return camera_height + ratio * (top_height - camera_height)


def _fuse(key: str, obj: Mapping[str, Any], sightings: list[_Sighting]) -> GroundedObject:
    w, h, d = (float(v) for v in obj["size"])
    widths = [s.width for s in sightings if s.width is not None]
    width = _clamp(float(np.median(widths)), 0.5 * w, 2.0 * w) if widths else w
    depth = _clamp(d * width / w, 0.5 * d, 1.5 * d)
    heights = [_top_edge(s, s.top_height, depth) for s in sightings if s.top_height is not None]
    height = _clamp(float(np.median(heights)), 0.5 * h, 2.0 * h) if heights else h
    centres = np.array([s.floor + depth / 2 * s.away for s in sightings])
    position = np.median(centres, axis=0)
    return GroundedObject(
        id=key,
        category=str(obj["category"])[:40] or "object",
        label=str(obj["label"])[:120],
        solid=bool(obj["solid"]),
        parts=copy.deepcopy(obj["parts"]),
        position=(float(position[0]), float(position[1])),
        size=(width, height, depth),
        confidence=min(MAX_OBJECT_CONFIDENCE, 0.3 + 0.15 * len(sightings)),
        detections=len(sightings),
    )


def ground_objects(
    analysis: Mapping[str, Any], views: Mapping[int, PosedView]
) -> list[GroundedObject]:
    """Floor objects placed from every frame that shows their bottom edge in view.

    The agent's size is the prior: measured widths, depths and heights are kept within a
    factor of two of it. Objects that no frame grounds are left out; ``floor_objects``
    lists what the agent reported, so the caller can count the drops.
    """
    agent = floor_objects(analysis)
    sightings: dict[str, list[_Sighting]] = {key: [] for key in agent}
    for entry in analysis["reply"]["frames"]:
        view = views.get(int(entry["frame"]))
        if view is None:
            continue
        for detection in entry.get("objects", []):
            if detection["id"] not in agent:
                continue
            sighting = _sight(view, detection)
            if sighting is not None:
                sightings[detection["id"]].append(sighting)
    return [_fuse(key, agent[key], found) for key, found in sightings.items() if found]
