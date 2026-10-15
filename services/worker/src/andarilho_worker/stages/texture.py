"""``texturing`` (CPU): colour the reviewed layout from the capture's video frames.

The walkable model is the layout itself: walls, floors, ceilings, openings and
obstacles as plain planar surfaces. This stage only paints them. Each registered frame
of the run is a camera with known intrinsics and a pose in the canonical world (the
run's ``worldFromColmap`` from ``normalizing``, scaled about the origin by the layout's
calibration), so every surface texel is blended from the frames that see it
(``geometry.texture``). The result is a GLB with unlit materials, written under this
attempt's prefix for ``packaging`` to publish.

Layout saves run this stage again on the revision's own layout. The frames and poses
come from the run's succeeded run-stage jobs, which every revision of a scene shares:
a layout job carries the run ID but not the run's outputs. A synthetic run has no
frames and gets the deterministic patterns of ``procedural_texture_layout``.
"""

from __future__ import annotations

import tarfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from andarilho_worker.errors import StageError
from andarilho_worker.geometry.colmap import read_cameras_bin, read_images_bin
from andarilho_worker.geometry.glb import write_textured_glb
from andarilho_worker.geometry.texture import (
    DEFAULT_PARAMS,
    TextureFrame,
    TextureParams,
    camera_in_world,
    procedural_texture_layout,
    texture_layout,
)
from andarilho_worker.stages.base import StageContext, StageResult, finish

MODEL_TYPE = "model/gltf-binary"
JPEG_QUALITY = 85  # the default of write_textured_glb
PROGRESS_STEPS = 10  # progress is reported about once per tenth of the blend's passes


def _load(conn: Any, ctx: StageContext, revision_id: str) -> dict[str, Any]:
    revision = conn.execute(
        "SELECT id, layout FROM scene_revisions WHERE id = %s AND project_id = %s",
        (revision_id, ctx.project_id),
    ).fetchone()
    rows = conn.execute(
        "SELECT stage, result_manifest FROM jobs WHERE run_id = %s AND project_id = %s "
        "AND scene_revision_id IS NULL AND state = 'succeeded' "
        "AND stage IN ('preparing', 'camera_reconstruction', 'normalizing')",
        (ctx.run_id, ctx.project_id),
    ).fetchall()
    stages = {row["stage"]: (row["result_manifest"] or {}).get("outputs") or {} for row in rows}
    return {"revision": revision, "stages": stages}


def _in_run(ctx: StageContext, key: str) -> bool:
    return key.startswith(f"tenants/{ctx.project_id}/runs/{ctx.run_id}/")


def _settings(ctx: StageContext) -> tuple[TextureParams, int]:
    config = dict(ctx.params.get("texture") or {})
    params = TextureParams(
        texel_size=float(config.get("texelSize", DEFAULT_PARAMS.texel_size)),
        page_size=int(config.get("pageSize", DEFAULT_PARAMS.page_size)),
        max_pages=int(config.get("maxPages", DEFAULT_PARAMS.max_pages)),
    )
    return params, int(config.get("jpegQuality", JPEG_QUALITY))


def _required(row: Mapping[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is None:
        raise StageError("MISSING_INPUT", details={"missing": key})
    return value


def _run_inputs(
    ctx: StageContext, stages: Mapping[str, Mapping[str, Any]]
) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
    """The dataset key, camera and normalizing outputs of a real capture, all checked up front.

    Checked before any frame is read, so a missing input fails in seconds, not after a
    full blend.
    """
    camera = stages.get("camera_reconstruction")
    normalizing = stages.get("normalizing")
    if normalizing is not None and "worldFromColmap" not in normalizing:
        # A scene from the splat pipeline has no camera-frame transform to texture from.
        raise StageError("SCENE_OUTDATED")
    if camera is None or normalizing is None:
        raise StageError("MISSING_INPUT", details={"missing": "run outputs"})
    dataset_key = str(_required(camera, "datasetKey"))
    if not _in_run(ctx, dataset_key):
        raise StageError("MISSING_INPUT", details={"missing": "dataset"})
    _required(camera, "coverage")
    _required(normalizing, "worldFromColmap")
    _required(normalizing, "scaleStatus")
    return dataset_key, camera, normalizing


def _loader(path: Path) -> Callable[[], np.ndarray]:
    def load() -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))

    return load


def _frames(
    ctx: StageContext, dataset_key: str, world: Sequence[float], render_scale: float
) -> list[TextureFrame]:
    """Every registered frame of the run's dataset, posed in the canonical world."""
    if ctx.store.head(dataset_key) is None:
        raise StageError("MISSING_INPUT", details={"missing": "dataset"})
    work = ctx.scratch / "texture"
    work.mkdir(parents=True, exist_ok=True)
    tar_path = work / "dataset.tar"
    ctx.store.download(dataset_key, tar_path)
    # The archive is written by this worker's reconstruction; the data filter still
    # refuses absolute paths and links that leave the directory.
    with tarfile.open(tar_path) as tar:
        tar.extractall(work, filter="data")
    tar_path.unlink()
    sparse = work / "sparse" / "0"
    try:
        cameras = read_cameras_bin(sparse / "cameras.bin")
        images = read_images_bin(sparse / "images.bin")
    except (OSError, ValueError) as exc:
        raise StageError("CAMERA_REGISTRATION_FAILED") from exc
    frames: list[TextureFrame] = []
    for image in images:
        camera = cameras.get(image.camera_id)
        path = work / "images" / image.name
        if camera is None or not path.is_file():
            raise StageError("MISSING_INPUT", details={"missing": "frame"})
        try:
            intrinsics = camera.intrinsics()
        except ValueError as exc:  # the frames were undistorted to PINHOLE cameras
            raise StageError("CAMERA_REGISTRATION_FAILED") from exc
        R, t = camera_in_world(image.rotation(), np.asarray(image.tvec), world, render_scale)
        frames.append(
            TextureFrame(image.name, intrinsics, R, t, camera.width, camera.height, _loader(path))
        )
    if not frames:
        raise StageError("CAMERA_REGISTRATION_FAILED")
    return frames


def _reporter(ctx: StageContext) -> Callable[[int, int], None]:
    """Progress for ``texture_layout``: checks cancellation on every call, reports sparsely."""

    def report(done: int, total: int) -> None:
        ctx.cancellation.raise_if_cancelled()
        if done % max(1, total // PROGRESS_STEPS) == 0:
            ctx.progress(
                "Texturing the model from the video.", current=done, total=total, unit="steps"
            )

    return report


def run(ctx: StageContext) -> StageResult:
    revision_id = str(ctx.output("sceneRevisionId"))
    data = ctx.query(lambda conn: _load(conn, ctx, revision_id))
    if data["revision"] is None:
        raise StageError("MISSING_INPUT")
    return paint(ctx, dict(data["revision"]["layout"]), data["stages"])


def paint(
    ctx: StageContext, layout: dict[str, Any], stages: Mapping[str, Mapping[str, Any]]
) -> StageResult:
    """Texture ``layout`` from the run's stage outputs, keyed by stage name.

    The database-free half of the stage, which the smoke runner calls directly.
    """
    started = time.monotonic()
    params, jpeg_quality = _settings(ctx)
    preparing = stages.get("preparing") or {}
    poster_key = preparing.get("posterKey") or None
    if poster_key is not None and not _in_run(ctx, str(poster_key)):
        raise StageError("MISSING_INPUT", details={"missing": "poster"})
    normalizing = stages.get("normalizing") or {}
    outputs: dict[str, Any] = {}
    if ctx.recipe.get("quality") == "synthetic":
        ctx.progress("Painting the layout's surfaces.")
        mesh = procedural_texture_layout(layout, params)
        frames = 0
        outputs["scaleStatus"] = normalizing.get("scaleStatus", "user_measured")
    else:
        dataset_key, camera, normalizing = _run_inputs(ctx, stages)
        calibration = layout.get("calibration")
        render_scale = float(calibration["renderScale"]) if calibration else 1.0
        ctx.progress("Reading the capture's frames and poses.")
        frame_list = _frames(ctx, dataset_key, normalizing["worldFromColmap"], render_scale)
        frames = len(frame_list)
        mesh = texture_layout(layout, frame_list, params, progress=_reporter(ctx))
        outputs["coverage"] = camera["coverage"]
        outputs["scaleStatus"] = normalizing["scaleStatus"]
    ctx.progress("Writing the textured model.")
    blob = write_textured_glb(mesh, jpeg_quality=jpeg_quality)
    key = ctx.attempt_key("model.glb")
    ctx.store.put_bytes(key, blob, MODEL_TYPE)
    coverage = {kind: round(float(value), 4) for kind, value in mesh.coverage.items()}
    outputs |= {
        "modelKey": key,
        "textureCoverage": coverage,
        "frameWarnings": list(normalizing.get("frameWarnings") or []),
    }
    if poster_key is not None:
        outputs["posterKey"] = str(poster_key)
    return finish(
        ctx,
        {
            "coverage": coverage,
            "frames": frames,
            "texelSize": mesh.texel_size,
            "pages": len(mesh.pages),
            "modelBytes": len(blob),
            "seconds": round(time.monotonic() - started, 1),
        },
        outputs=outputs,
    )
