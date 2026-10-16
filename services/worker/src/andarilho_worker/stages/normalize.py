"""``normalizing``: the canonical frame from COLMAP's camera poses and sparse points.

Transform chain (README section 9): COLMAP's sparse model has its own world, with an
arbitrary up and scale. ``frame.frame_from_sparse`` levels it so that +Y is up, turns the
walls onto the X and Z axes, scales it so the cameras sit at a walking person's height
(1.5 m, an estimate until the user measures a distance) and puts the floor at y = 0.
The result is recorded as ``worldFromColmap`` (column major,
``THREE.Matrix4.fromArray`` compatible). Points and poses are never rewritten: the
texturing stage applies the matrix when it needs them in the canonical world.
Scale is ``estimated`` until a user measurement exists.

The stage streams the ``sparse/`` members of ``dataset.tar`` (the frames under
``images/`` are never extracted) and writes ``evidence.json``: the reliable sparse
points and the camera centres, in the canonical world, for the review editor to draw
under the walls.

The synthetic recipe has no capture. Its scene is the bundled two-room layout, already
in the canonical world, so it writes no files and records no frame.
"""

from __future__ import annotations

import json
import tarfile
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from andarilho_worker.errors import StageError
from andarilho_worker.geometry.colmap import (
    ColmapImage,
    SparsePoints,
    read_images_bin,
    read_points3d_full,
)
from andarilho_worker.geometry.frame import FrameEstimate, frame_from_sparse, reliable_points
from andarilho_worker.stages.base import StageContext, StageResult, finish

# The evidence is drawn by the editor, which only needs the shape of the capture.
EVIDENCE_POINTS = 5000
EVIDENCE_DECIMALS = 3  # millimetres
EVIDENCE_SCHEMA_VERSION = 1


def synthetic_layout() -> dict[str, Any]:
    text = resources.files("andarilho_worker.data").joinpath("synthetic-two-room.layout.json")
    return json.loads(text.read_text("utf-8"))


def apply_matrix(m: list[float], points: np.ndarray) -> np.ndarray:
    mat = np.array(m, dtype=np.float64).reshape(4, 4).T  # column-major -> row-major
    homo = np.column_stack([points, np.ones(len(points))])
    return (homo @ mat.T)[:, :3]


def _extract_sparse(ctx: StageContext, dataset_key: str) -> Path:
    """Stream the dataset archive to scratch and extract only its ``sparse/`` members.

    The archive can hold hundreds of megabytes of frames, which normalizing never needs.
    The archive is removed once its model is out, so scratch use stays bounded.
    """
    if ctx.store.head(dataset_key) is None:
        raise StageError("MISSING_INPUT", details={"missing": "dataset"})
    archive = ctx.scratch / "dataset.tar"
    ctx.store.download(dataset_key, archive)
    work = ctx.scratch / "normalize"
    work.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.startswith("sparse/")]
            tar.extractall(work, members=members, filter="data")
    except (tarfile.TarError, EOFError) as exc:
        raise StageError("CAMERA_REGISTRATION_FAILED") from exc
    finally:
        archive.unlink(missing_ok=True)
    return work / "sparse" / "0"


def _read_model(model: Path) -> tuple[list[ColmapImage], SparsePoints]:
    try:
        images = read_images_bin(model / "images.bin")
        points = read_points3d_full(model / "points3D.bin")
    except (OSError, ValueError) as exc:
        raise StageError("CAMERA_REGISTRATION_FAILED") from exc
    return images, points


def _rounded(values: np.ndarray) -> list[list[float]]:
    return np.round(values, EVIDENCE_DECIMALS).tolist()


def capture_evidence(
    frame: FrameEstimate, images: list[ColmapImage], points: SparsePoints, *, seed: int
) -> tuple[bytes, int]:
    """``evidence.json`` for the review editor, and the number of points it holds.

    Points and camera centres are in the canonical world, in metres. The points are a
    seeded subsample of the reliable ones, sorted back into their original order.
    """
    xyz = reliable_points(points)
    if len(xyz) > EVIDENCE_POINTS:
        chosen = np.random.default_rng(seed).choice(len(xyz), EVIDENCE_POINTS, replace=False)
        xyz = xyz[np.sort(chosen)]
    walk = sorted(images, key=lambda image: image.name)
    centres = np.array([image.center() for image in walk], dtype=np.float64).reshape(-1, 3)
    evidence = {
        "schemaVersion": EVIDENCE_SCHEMA_VERSION,
        "units": "m",
        "points": _rounded(apply_matrix(frame.matrix, xyz)),
        "cameras": _rounded(apply_matrix(frame.matrix, centres)),
    }
    blob = json.dumps(evidence, separators=(",", ":"), allow_nan=False).encode()
    return blob, len(xyz)


def run(ctx: StageContext) -> StageResult:
    if ctx.outputs().get("quality") == "synthetic":
        return finish(
            ctx,
            {"synthetic": True},
            outputs={"scaleStatus": "user_measured", "proposalBounds": None},
        )
    ctx.progress("Reading the camera poses and sparse points.")
    model = _extract_sparse(ctx, str(ctx.output("datasetKey")))
    images, points = _read_model(model)
    seed = int(ctx.params.get("seed", 0))
    ctx.progress("Finding the floor and the walking height.")
    try:
        frame = frame_from_sparse(points, images, seed=seed)
    except ValueError as exc:
        raise StageError("CAMERA_REGISTRATION_FAILED") from exc
    blob, evidence_points = capture_evidence(frame, images, points, seed=seed)
    evidence_key = ctx.attempt_key("evidence.json")
    ctx.store.put_bytes(evidence_key, blob, "application/json")
    result: dict[str, Any] = {
        "frame": frame.summary(),
        "sparsePoints": int(len(points.xyz)),
        "cameras": len(images),
        "evidencePoints": evidence_points,
    }
    outputs: dict[str, Any] = {
        "worldFromColmap": frame.matrix,
        "scaleStatus": "estimated",
        "proposalBounds": {
            "minXZ": frame.min_xz,
            "maxXZ": frame.max_xz,
            "floorY": 0.0,
            "height": frame.height,
        },
        "proposalSpawn": {"feet": frame.spawn_feet, "yawRadians": frame.spawn_yaw},
        "frameWarnings": frame.warnings,
        "evidenceKey": evidence_key,
    }
    return finish(ctx, result, outputs=outputs)
