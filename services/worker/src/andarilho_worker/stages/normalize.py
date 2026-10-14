"""``normalizing``: clean and recenter the exported splats; produce the render asset.

Transform chain (README section 9): the exported PLY is in Nerfstudio's
dataparser frame (Z up, poses scaled into a unit cube). Recentering is *baked*
into the splat positions (a pure translation needs no orientation/covariance/SH
update). The rest is recorded as ``worldFromDecodedRender`` (column major,
``THREE.Matrix4.fromArray`` compatible) and applied by the viewer at runtime: a
proper rotation that levels the floor found under the camera path and turns Z up
into Y up, a uniform scale that puts the cameras at a walking person's height
(nominal meters), and a translation that puts that floor at y = 0
(``geometry.frame``). Scale is ``estimated`` until a user measurement exists.
Runs without a recorded camera path keep the plain Z-up -> Y-up rotation.

The synthetic recipe instead renders a splat cloud directly in the canonical
world frame from the bundled synthetic layout, so its matrix is identity.
"""

from __future__ import annotations

import json
import time
from importlib import resources
from typing import Any

import numpy as np

from andarilho_worker.artifacts import MAX_RENDER_BYTES
from andarilho_worker.errors import StageError
from andarilho_worker.geometry.frame import (
    CameraPath,
    FrameEstimate,
    colmap_to_export,
    estimate_frame,
)
from andarilho_worker.geometry.ply import (
    CleanedPly,
    PlyError,
    clean_splat_ply_file,
    write_splat_ply,
)
from andarilho_worker.geometry.synthetic import synthetic_splats
from andarilho_worker.manifest import IDENTITY_MATRIX
from andarilho_worker.recipes import render_argv
from andarilho_worker.runner import ProcessFailed, ProcessTimeout, tool_available
from andarilho_worker.stages.base import StageContext, StageResult, finish

# Nerfstudio/COLMAP Z-up decoded frame -> Andarilho world (+Y up): (x, y, z) -> (x, z, -y)
Z_UP_TO_Y_UP = [1.0, 0, 0, 0, 0, 0, -1.0, 0, 0, 1.0, 0, 0, 0, 0, 0, 1.0]
MIN_SPLATS = 1000
BOUNDS_SAMPLE = 2_000_000
# SOG compression on the CPU (splat-transform's k-means, one core) is slow for the
# higher spherical-harmonic bands, which the recipe drops first (renderShDegree). It
# gets at most this long, and half the stage's remaining time, before the scene falls
# back to its PLY.
SOG_MAX_SECONDS = 900
# Splatfacto's default; a recipe without renderShDegree keeps every band.
TRAINED_SH_DEGREE = 3
# Only confident splats place the floor: translucent ones are mostly floaters.
FLOOR_MIN_OPACITY = 0.8


def synthetic_layout() -> dict[str, Any]:
    text = resources.files("andarilho_worker.data").joinpath("synthetic-two-room.layout.json")
    return json.loads(text.read_text("utf-8"))


def apply_matrix(m: list[float], points: np.ndarray) -> np.ndarray:
    mat = np.array(m, dtype=np.float64).reshape(4, 4).T  # column-major -> row-major
    homo = np.column_stack([points, np.ones(len(points))])
    return (homo @ mat.T)[:, :3]


def camera_frame(ctx: StageContext, cleaned: CleanedPly) -> FrameEstimate | None:
    """The canonical frame from the recorded camera path, or None for older runs."""
    outputs = ctx.outputs()
    if not outputs.get("camerasKey") or not outputs.get("dataparserTransformsKey"):
        return None
    record = json.loads(ctx.store.get_bytes(str(outputs["camerasKey"])))
    dataparser = json.loads(ctx.store.get_bytes(str(outputs["dataparserTransformsKey"])))
    try:
        path = CameraPath.from_record(record["cameras"])
        transform, scale = dataparser["transform"], float(dataparser["scale"])
        exported = CameraPath(
            colmap_to_export(path.centers, transform, scale, cleaned.center),
            colmap_to_export(path.forwards, transform, scale, cleaned.center, directions=True),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StageError("EXPORT_INVALID", "The camera path could not be read.") from exc
    if len(exported.centers) == 0 or not (scale > 0 and np.isfinite(exported.centers).all()):
        raise StageError("EXPORT_INVALID", "The camera path could not be read.")
    opaque = cleaned.positions[cleaned.opacity > FLOOR_MIN_OPACITY]
    if len(opaque) < MIN_SPLATS:
        opaque = cleaned.positions
    return estimate_frame(opaque, exported, seed=int(ctx.params.get("seed", 0)))


def run(ctx: StageContext) -> StageResult:
    if ctx.outputs().get("quality") == "synthetic":
        blob = write_splat_ply(
            synthetic_splats(synthetic_layout()), comments=["andarilho synthetic"]
        )
        key = ctx.attempt_key("scene.ply")
        ctx.store.put_bytes(key, blob, "application/octet-stream")
        return finish(
            ctx,
            {"splats": "synthetic", "bytes": len(blob)},
            outputs={
                "renderKey": key,
                "renderFormat": "ply",
                "worldFromDecodedRender": IDENTITY_MATRIX,
                "scaleStatus": "user_measured",
                "proposalBounds": None,
            },
        )
    ply_key = str(ctx.output("plyKey"))
    exported = ctx.store.head(ply_key)
    if exported is None:
        raise StageError("MISSING_INPUT", details={"missing": "exported splats"})
    if exported.size > MAX_RENDER_BYTES:
        raise StageError("BUDGET_EXHAUSTED", "The exported scene is too large to package.")
    export = ctx.scratch / "export.ply"
    # Streamed to scratch and cleaned chunk by chunk: exports can be several GiB.
    ctx.store.download(ply_key, export)
    work = ctx.scratch / "normalize"
    work.mkdir(parents=True, exist_ok=True)
    ply_path = work / "normalized.ply"
    try:
        cleaned = clean_splat_ply_file(
            export, ply_path, comments=["andarilho normalized (recentered)"]
        )
    except PlyError as exc:
        raise StageError("EXPORT_INVALID") from exc
    export.unlink()  # the archive below replaces it; keep scratch use bounded
    if cleaned.count < MIN_SPLATS:
        raise StageError("EXPORT_INVALID", "Too few valid splats were produced.")
    archive_key = ctx.attempt_key("normalized.ply")
    ctx.store.put_file(archive_key, ply_path, "application/octet-stream")
    render_key, render_format = archive_key, "ply"
    compression: dict[str, Any] = {"format": "ply", "reason": "no compressor"}
    if tool_available("splat-transform"):
        sog_path = work / "scene.sog"
        sh_degree = int(ctx.params.get("renderShDegree", TRAINED_SH_DEGREE))
        budget = min(SOG_MAX_SECONDS, ctx.remaining_seconds() / 2)
        started = time.monotonic()
        try:
            ctx.progress("Compressing the render asset.")
            argv = render_argv(
                "splat_transform", input=ply_path, sh_degree=sh_degree, output=sog_path
            )
            ctx.run(argv, timeout=budget)
            render_key, render_format = ctx.attempt_key("scene.sog"), "sog"
            ctx.store.put_file(render_key, sog_path, "application/octet-stream")
            compression = {
                "format": "sog",
                "bytes": sog_path.stat().st_size,
                "shDegree": sh_degree,
            }
        except (ProcessFailed, ProcessTimeout) as exc:
            # Keep the PLY: larger to deliver, but a slow or failed compression never
            # costs the user their reconstruction.
            reason = "timeout" if isinstance(exc, ProcessTimeout) else "failed"
            compression = {"format": "ply", "reason": reason}
        compression["seconds"] = round(time.monotonic() - started, 1)
    ctx.progress("Finding the floor and the walking height.")
    frame = camera_frame(ctx, cleaned)
    result: dict[str, Any] = {
        "splats": cleaned.count,
        "dropped": cleaned.dropped,
        "recenteredBy": cleaned.center,
        "compression": compression,
    }
    outputs: dict[str, Any] = {
        "renderKey": render_key,
        "renderFormat": render_format,
        "archivePlyKey": archive_key,
        "scaleStatus": "estimated",
    }
    if frame is not None:
        result["frame"] = frame.summary()
        outputs |= {
            "worldFromDecodedRender": frame.matrix,
            "proposalBounds": {
                "minXZ": frame.min_xz,
                "maxXZ": frame.max_xz,
                "floorY": 0.0,
                "height": frame.height,
            },
            "proposalSpawn": {"feet": frame.spawn_feet, "yawRadians": frame.spawn_yaw},
            "frameWarnings": frame.warnings,
        }
        return finish(ctx, result, outputs=outputs)
    # Percentile bounds from an evenly strided sample keep this step small for huge clouds.
    sample = cleaned.positions[:: max(1, cleaned.count // BOUNDS_SAMPLE)]
    world = apply_matrix(Z_UP_TO_Y_UP, sample)
    lo = np.percentile(world, 5, axis=0)
    hi = np.percentile(world, 95, axis=0)
    floor_y = float(np.percentile(world[:, 1], 2))
    ceiling = float(np.percentile(world[:, 1], 98))
    outputs |= {
        "worldFromDecodedRender": Z_UP_TO_Y_UP,
        "proposalBounds": {
            "minXZ": [float(lo[0]), float(lo[2])],
            "maxXZ": [float(hi[0]), float(hi[2])],
            "floorY": floor_y,
            "height": max(2.2, min(4.0, ceiling - floor_y)),
        },
    }
    return finish(ctx, result, outputs=outputs)
