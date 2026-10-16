"""``scene_analysis`` (CPU): rooms, doors and furniture recognised by an AI model.

Only when ``AI_SCENE_ANALYSIS`` is on does anything leave the worker. Then a few keyframes
of the walkthrough, a few registered photos and the floor plan are sent, downscaled to
JPEG, to the configured OpenAI-compatible model (OpenCode by default). Camera poses,
storage keys and the rest of the capture stay with the worker. The model's reply is
checked by :func:`andarilho_worker.analysis.normalise_reply` and written to
``analysis.json`` for ``geometry_proposal``, which builds the rooms, doors and furniture
into the layout.

A provider failure never fails the run: the artifact records why, and the proposal falls
back to a simple room with a warning. The synthetic recipe, and runs with the switch off,
make no call and write no artifact.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from andarilho_worker.ai import AIInvalidReply, AIUnavailable, VisionClient, session_id
from andarilho_worker.analysis import (
    SYSTEM_PROMPT,
    build_content,
    data_url,
    encode_image,
    floor_plan_image,
    frame_kind,
    normalise_reply,
    select_keyframes,
)
from andarilho_worker.errors import StageError
from andarilho_worker.geometry.colmap import ColmapCamera, read_cameras_bin, read_images_bin
from andarilho_worker.geometry.texture import camera_in_world
from andarilho_worker.recipes import COMMON
from andarilho_worker.stages.base import StageContext, StageResult, finish

ARTIFACT_SCHEMA_VERSION = 1
FLOOR_Y = 0.0  # the canonical floor, where normalizing puts it
MAX_FLOOR_PLAN_BYTES = 20 * 1024**2  # the upload limit for a floor plan
UNAVAILABLE_WARNING = (
    "Rooms, doors and furniture could not be recognised automatically because the AI service "
    "was unavailable; the scene shows a simple room instead."
)
INVALID_WARNING = (
    "Rooms, doors and furniture could not be recognised automatically; the scene shows a simple "
    "room instead."
)


@dataclass(frozen=True)
class _View:
    """A registered image of the capture, posed in the canonical world."""

    name: str
    camera: ColmapCamera
    rotation: np.ndarray  # (3, 3), world to camera
    translation: np.ndarray  # (3,)
    centre: np.ndarray  # (3,), where the camera stands
    heading: float  # radians; the front direction is (sin, cos) in x and z


@dataclass(frozen=True)
class _Outcome:
    status: str  # complete | unavailable | invalid
    reason: str | None
    reply: dict[str, Any] | None
    warnings: list[str]
    usage: dict[str, int]


def _synthetic(ctx: StageContext) -> bool:
    return ctx.recipe.get("quality") == "synthetic" or ctx.outputs().get("quality") == "synthetic"


def _outputs(status: str, key: str | None, warnings: list[str]) -> dict[str, Any]:
    outputs: dict[str, Any] = {"analysisStatus": status, "analysisWarnings": warnings}
    if key is not None:
        outputs["analysisKey"] = key
    return outputs


def _poses(archive: Path, work: Path, world: list[float]) -> list[_View]:
    """Every registered image in walk order, posed in the canonical world.

    Only the archive's ``sparse/`` members are extracted: the frames are not needed for
    the poses, and the selected ones are extracted later.
    """
    views: list[_View] = []
    try:
        with tarfile.open(archive) as tar:
            sparse = [m for m in tar.getmembers() if m.isfile() and m.name.startswith("sparse/")]
            tar.extractall(work, members=sparse, filter="data")
        model = work / "sparse" / "0"
        cameras = read_cameras_bin(model / "cameras.bin")
        images = read_images_bin(model / "images.bin")
        for image in sorted(images, key=lambda item: item.name):
            camera = cameras.get(image.camera_id)
            if camera is None:
                raise StageError("MISSING_INPUT", details={"missing": "camera"})
            rotation, translation = camera_in_world(
                image.rotation(), np.asarray(image.tvec), world, 1.0
            )
            forward = rotation[2]  # the optical axis, as a world direction
            views.append(
                _View(
                    image.name,
                    camera,
                    rotation,
                    translation,
                    -rotation.T @ translation,
                    float(np.arctan2(forward[0], forward[2])),
                )
            )
    except (OSError, tarfile.TarError, EOFError, ValueError) as exc:
        raise StageError("CAMERA_REGISTRATION_FAILED") from exc
    if not views:
        raise StageError("CAMERA_REGISTRATION_FAILED")
    return views


def _extract_frames(archive: Path, work: Path, names: list[str]) -> None:
    """Extract only the selected frames of the archive, under ``images/``."""
    wanted = {f"images/{name}" for name in names}
    try:
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name in wanted]
            if len(members) != len(wanted):
                raise StageError("MISSING_INPUT", details={"missing": "frame"})
            tar.extractall(work, members=members, filter="data")
    except (OSError, tarfile.TarError, EOFError) as exc:
        raise StageError("CAMERA_REGISTRATION_FAILED") from exc


def _intrinsics(camera: ColmapCamera) -> list[float]:
    """fx, fy, cx, cy of an undistorted (pinhole) dataset camera."""
    try:
        matrix = camera.intrinsics()
    except ValueError as exc:
        raise StageError("CAMERA_REGISTRATION_FAILED") from exc
    return [float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[0, 2]), float(matrix[1, 2])]


def _floor_plan_jpeg(ctx: StageContext, long_edge: int, jpeg_quality: int) -> bytes | None:
    """The floor plan as a JPEG for the model, or None when it is absent or unusable.

    The plan must sit under this project's keys, be within the upload limit and still hash
    to the value validation recorded. Anything else is skipped: the analysis does not need
    the plan to produce a reply.
    """
    plan = ctx.outputs().get("floorPlan")
    if not plan:
        return None
    key = str(plan.get("key") or "")
    if not key.startswith(f"tenants/{ctx.project_id}/"):
        return None
    info = ctx.store.head(key)
    if info is None or info.size > MAX_FLOOR_PLAN_BYTES:
        return None
    data = ctx.store.get_bytes(key, max_bytes=MAX_FLOOR_PLAN_BYTES)
    if hashlib.sha256(data).hexdigest() != plan.get("sha256"):
        return None
    image = floor_plan_image(data, str(plan.get("contentType") or ""))
    if image is None:
        return None
    jpeg, _, _ = encode_image(image, long_edge, jpeg_quality)
    return jpeg


def _ask(ctx: StageContext, content: list[dict[str, Any]], frames: set[int]) -> _Outcome:
    """One call to the model, with its failures reduced to the artifact's status."""
    client = ctx.services.vision or VisionClient.from_settings(ctx.services.settings)
    try:
        completion = client.complete_json(SYSTEM_PROMPT, content, session=session_id(ctx.run_id))
        reply = normalise_reply(completion.data, frames)
    except AIUnavailable as exc:
        return _Outcome("unavailable", exc.reason, None, [UNAVAILABLE_WARNING], {})
    except AIInvalidReply:
        return _Outcome("invalid", "bad_reply", None, [INVALID_WARNING], {})
    return _Outcome("complete", None, reply, [], completion.usage)


def run(ctx: StageContext) -> StageResult:
    settings = ctx.services.settings
    if _synthetic(ctx):
        return finish(ctx, {"status": "skipped"}, outputs=_outputs("skipped", None, []))
    if not settings.ai_scene_analysis:
        return finish(ctx, {"status": "disabled"}, outputs=_outputs("disabled", None, []))
    started = time.monotonic()
    config = {**COMMON["analysis"], **(ctx.params.get("analysis") or {})}
    dataset_key = str(ctx.output("datasetKey"))
    if not dataset_key.startswith(f"tenants/{ctx.project_id}/runs/{ctx.run_id}/"):
        raise StageError("MISSING_INPUT", details={"missing": "dataset"})
    world = [float(value) for value in ctx.output("worldFromColmap")]
    if ctx.store.head(dataset_key) is None:
        raise StageError("MISSING_INPUT", details={"missing": "dataset"})

    ctx.progress("Choosing frames for the scene analysis.")
    work = ctx.scratch / "analysis"
    work.mkdir(parents=True, exist_ok=True)
    archive = work / "dataset.tar"
    ctx.store.download(dataset_key, archive)
    try:
        views = _poses(archive, work, world)
        chosen = select_keyframes(
            [
                (view.name, frame_kind(view.name), (view.centre[0], view.centre[2]), view.heading)
                for view in views
            ],
            count=int(config["keyframes"]),
            max_photos=int(config["maxPhotos"]),
        )
        _extract_frames(archive, work, [views[index].name for index in chosen])
    finally:
        archive.unlink(missing_ok=True)

    long_edge, jpeg_quality = int(config["imageLongEdge"]), int(config["jpegQuality"])
    keyframes: list[dict[str, Any]] = []
    pairs: list[tuple[int, str]] = []
    image_bytes = 0
    for index in chosen:
        ctx.cancellation.raise_if_cancelled()
        view = views[index]
        jpeg, _, _ = encode_image(work / "images" / view.name, long_edge, jpeg_quality)
        image_bytes += len(jpeg)
        pairs.append((index, data_url(jpeg)))
        keyframes.append(
            {
                "frame": index,
                "name": view.name,
                "kind": frame_kind(view.name),
                "width": view.camera.width,
                "height": view.camera.height,
                "intrinsics": _intrinsics(view.camera),
                "rotation": view.rotation.reshape(-1).tolist(),
                "translation": view.translation.tolist(),
            }
        )
    floor_jpeg = _floor_plan_jpeg(ctx, int(config["floorPlanLongEdge"]), jpeg_quality)
    floor_url = None
    if floor_jpeg is not None:
        image_bytes += len(floor_jpeg)
        floor_url = data_url(floor_jpeg)
    photo_frames = [entry["frame"] for entry in keyframes if entry["kind"] == "photo"]
    content = build_content(pairs, floor_url, photo_frames)

    ctx.progress("Asking the AI model about rooms, doors and furniture.", keyframes=len(pairs))
    ctx.cancellation.raise_if_cancelled()
    outcome = _ask(ctx, content, set(chosen))
    ctx.cancellation.raise_if_cancelled()

    artifact = {
        "schemaVersion": ARTIFACT_SCHEMA_VERSION,
        "status": outcome.status,
        "reason": outcome.reason,
        "model": settings.ai_model,
        "floorY": FLOOR_Y,
        "keyframes": keyframes,
        "reply": outcome.reply,
    }
    blob = json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False)
    key = ctx.attempt_key("analysis.json")
    ctx.store.put_bytes(key, blob.encode(), "application/json")
    manifest = {
        "status": outcome.status,
        "reason": outcome.reason,
        "model": settings.ai_model,
        "keyframes": len(keyframes),
        "photos": len(photo_frames),
        "floorPlanSent": floor_jpeg is not None,
        "imageBytes": image_bytes,
        "usage": outcome.usage,
        "seconds": round(time.monotonic() - started, 1),
    }
    return finish(ctx, manifest, outputs=_outputs(outcome.status, key, outcome.warnings))
