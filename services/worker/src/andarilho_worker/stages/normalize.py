"""``normalizing``: clean and recenter the exported splats; produce the render asset.

Transform chain (README section 9): the exported PLY is in Nerfstudio's
dataparser frame (Z up). Recentering is *baked* into the splat positions (a pure
translation needs no orientation/covariance/SH update). The remaining
Z-up -> Y-up proper rotation is recorded as ``worldFromDecodedRender`` (column
major, ``THREE.Matrix4.fromArray`` compatible) and applied by the viewer at
runtime. Scale is ``estimated`` until a user measurement exists.

The synthetic recipe instead renders a splat cloud directly in the canonical
world frame from the bundled synthetic layout, so its matrix is identity.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

import numpy as np

from andarilho_worker.artifacts import MAX_RENDER_BYTES
from andarilho_worker.errors import StageError
from andarilho_worker.geometry.ply import (
    PlyError,
    drop_non_finite,
    read_splat_ply_file,
    recenter,
    write_splat_ply,
    write_splat_ply_file,
)
from andarilho_worker.geometry.synthetic import synthetic_splats
from andarilho_worker.manifest import IDENTITY_MATRIX
from andarilho_worker.recipes import render_argv
from andarilho_worker.runner import ProcessFailed, tool_available
from andarilho_worker.stages.base import StageContext, StageResult, finish

# Nerfstudio/COLMAP Z-up decoded frame -> Andarilho world (+Y up): (x, y, z) -> (x, z, -y)
Z_UP_TO_Y_UP = [1.0, 0, 0, 0, 0, 0, -1.0, 0, 0, 1.0, 0, 0, 0, 0, 0, 1.0]
MIN_SPLATS = 1000


def synthetic_layout() -> dict[str, Any]:
    text = resources.files("andarilho_worker.data").joinpath("synthetic-two-room.layout.json")
    return json.loads(text.read_text("utf-8"))


def apply_matrix(m: list[float], points: np.ndarray) -> np.ndarray:
    mat = np.array(m, dtype=np.float64).reshape(4, 4).T  # column-major -> row-major
    homo = np.column_stack([points, np.ones(len(points))])
    return (homo @ mat.T)[:, :3]


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
    # Streamed to scratch and memory-mapped: exports can exceed the inline object limit.
    ctx.store.download(ply_key, export)
    try:
        cloud = read_splat_ply_file(export)
    except PlyError as exc:
        raise StageError("EXPORT_INVALID") from exc
    cloud, dropped = drop_non_finite(cloud)
    if cloud.count < MIN_SPLATS:
        raise StageError("EXPORT_INVALID", "Too few valid splats were produced.")
    cloud, center = recenter(cloud)
    work = ctx.scratch / "normalize"
    work.mkdir(parents=True, exist_ok=True)
    ply_path = work / "normalized.ply"
    write_splat_ply_file(cloud, ply_path, comments=["andarilho normalized (recentered)"])
    export.unlink()  # the archive below replaces it; keep scratch use bounded
    archive_key = ctx.attempt_key("normalized.ply")
    ctx.store.put_file(archive_key, ply_path, "application/octet-stream")
    render_key, render_format = archive_key, "ply"
    if tool_available("splat-transform"):
        sog_path = work / "scene.sog"
        try:
            ctx.progress("Compressing the render asset.")
            ctx.run(render_argv("splat_transform", input=ply_path, output=sog_path), timeout=1800)
            render_key, render_format = ctx.attempt_key("scene.sog"), "sog"
            ctx.store.put_file(render_key, sog_path, "application/octet-stream")
        except ProcessFailed:
            render_key, render_format = archive_key, "ply"  # CPU fallback: keep the PLY
    world = apply_matrix(Z_UP_TO_Y_UP, cloud.positions())
    lo = np.percentile(world, 5, axis=0)
    hi = np.percentile(world, 95, axis=0)
    floor_y = float(np.percentile(world[:, 1], 2))
    ceiling = float(np.percentile(world[:, 1], 98))
    return finish(
        ctx,
        {"splats": cloud.count, "dropped": dropped, "recenteredBy": center},
        outputs={
            "renderKey": render_key,
            "renderFormat": render_format,
            "archivePlyKey": archive_key,
            "worldFromDecodedRender": Z_UP_TO_Y_UP,
            "scaleStatus": "estimated",
            "proposalBounds": {
                "minXZ": [float(lo[0]), float(lo[2])],
                "maxXZ": [float(hi[0]), float(hi[2])],
                "floorY": floor_y,
                "height": max(2.2, min(4.0, ceiling - floor_y)),
            },
        },
    )
