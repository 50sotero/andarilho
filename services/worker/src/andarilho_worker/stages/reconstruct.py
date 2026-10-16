"""``camera_reconstruction`` (GPU): COLMAP features, sequential matching, mapping, gates.

Coverage gates (README section 8, initial heuristics to calibrate on fixtures):
at least ``minRegisteredViews`` registered views and at least
``minRegisteredRatio`` of the selected views in the main component. A large
second component is reported as ``DISCONNECTED_CAPTURE``.
"""

from __future__ import annotations

import json
import shutil
import struct
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from andarilho_worker.errors import StageError, TransientError
from andarilho_worker.geometry.colmap import read_camera_poses, read_points3d
from andarilho_worker.recipes import render_argv
from andarilho_worker.runner import ProcessFailed
from andarilho_worker.stages.base import StageContext, StageResult, finish


@dataclass(frozen=True)
class SparseModel:
    path: Path
    registered: int
    points: int


def read_images_count(path: Path) -> int:
    with path.open("rb") as fh:
        data = fh.read(8)
    if len(data) != 8:
        raise ValueError("truncated images.bin")
    return struct.unpack("<Q", data)[0]


def list_models(sparse: Path) -> list[SparseModel]:
    models = []
    for child in sorted(p for p in sparse.iterdir() if p.is_dir()):
        images = child / "images.bin"
        points = child / "points3D.bin"
        if images.exists():
            n_points = 0
            if points.exists():
                with points.open("rb") as fh:
                    head = fh.read(8)
                n_points = struct.unpack("<Q", head)[0] if len(head) == 8 else 0
            models.append(SparseModel(child, read_images_count(images), n_points))
    return sorted(models, key=lambda m: m.registered, reverse=True)


def coverage_gate(
    models: list[SparseModel], selected: int, *, min_views: int, min_ratio: float
) -> dict[str, object]:
    """Raise a stable error code when coverage is insufficient; return quality otherwise."""
    if not models or models[0].registered == 0:
        raise StageError("CAMERA_REGISTRATION_FAILED")
    main = models[0]
    components = sum(1 for m in models if m.registered >= 3)
    ratio = main.registered / max(1, selected)
    quality = {
        "registeredViews": main.registered,
        "selectedViews": selected,
        "disconnectedComponents": components,
        "registeredRatio": round(ratio, 4),
    }
    second = models[1].registered if len(models) > 1 else 0
    if ratio < min_ratio and second >= max(5, int(0.1 * selected)):
        raise StageError("DISCONNECTED_CAPTURE", details=quality)
    if main.registered < min_views or ratio < min_ratio:
        raise StageError("INSUFFICIENT_OVERLAP", details=quality)
    return quality


def photo_pairs(names: list[str]) -> list[tuple[str, str]]:
    """Every supplementary photo paired with every other image (unordered, unique)."""
    photos = [n for n in names if n.startswith("photos")]
    pairs: set[tuple[str, str]] = set()
    for photo in photos:
        for other in names:
            if other != photo:
                pairs.add((min(photo, other), max(photo, other)))
    return sorted(pairs)


def match_features(
    ctx: StageContext, database: Path, names: list[str], work: Path, *, overlap: int
) -> None:
    """Sequential matching only where filenames are a real capture order (video frames).

    Photos are ordered by upload ID, not by capture, so photo-only sets use the
    exhaustive matcher and mixed sets add explicit photo-to-everything pairs.
    """
    has_video = any(not n.startswith("photos") for n in names)
    has_photos = any(n.startswith("photos") for n in names)
    if not has_video:
        ctx.run(render_argv("colmap_exhaustive_matcher", database=database))
        return
    ctx.run(render_argv("colmap_sequential_matcher", database=database, overlap=overlap))
    if has_photos:
        pairs_file = work / "photo-pairs.txt"
        pairs_file.write_text("".join(f"{a} {b}\n" for a, b in photo_pairs(names)))
        ctx.run(render_argv("colmap_matches_importer", database=database, pairs=pairs_file))


def run(ctx: StageContext) -> StageResult:
    params = ctx.params
    prefix = ctx.output("framesPrefix")
    names = list(ctx.output("frameNames"))
    work = ctx.scratch / "colmap"
    images = work / "images"
    for name in names:
        ctx.cancellation.raise_if_cancelled()
        ctx.store.download(prefix + name, images / name)
    database = work / "database.db"
    sparse = work / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    long_edge = int(params.get("longEdge", 1600))
    try:
        ctx.progress("Extracting features.", images=len(names))
        ctx.run(
            render_argv(
                "colmap_feature_extractor", database=database, images=images, long_edge=long_edge
            )
        )
        ctx.progress("Matching features.")
        match_features(ctx, database, names, work, overlap=int(params.get("sequentialOverlap", 10)))
    except ProcessFailed as exc:
        # Extraction and matching succeed on any decodable frames; a failure here is the
        # toolchain or GPU, not the capture, so it must not tell the user to record again.
        raise TransientError() from exc
    try:
        ctx.progress("Estimating camera positions.")
        ctx.run(render_argv("colmap_mapper", database=database, images=images, sparse=sparse))
    except ProcessFailed as exc:
        raise StageError("CAMERA_REGISTRATION_FAILED") from exc
    quality = coverage_gate(
        list_models(sparse),
        len(names),
        min_views=int(params.get("minRegisteredViews", 30)),
        min_ratio=float(params.get("minRegisteredRatio", 0.8)),
    )
    main = list_models(sparse)[0]
    undistorted = work / "undistorted"
    try:
        ctx.progress("Undistorting images.", **quality)
        ctx.run(
            render_argv(
                "colmap_image_undistorter",
                images=images,
                model=main.path,
                output=undistorted,
                long_edge=long_edge,
            )
        )
    except ProcessFailed as exc:
        raise TransientError() from exc
    # The dataset keeps images/ for texturing and sparse/0/ for normalizing, which reads
    # the poses and points out of the archive without extracting the frames.
    dataset_sparse = undistorted / "sparse" / "0"
    dataset_sparse.mkdir(parents=True, exist_ok=True)
    for item in list((undistorted / "sparse").iterdir()):
        if item.is_file():
            shutil.move(str(item), dataset_sparse / item.name)
    points = read_points3d(dataset_sparse / "points3D.bin")
    tar_path = work / "dataset.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(undistorted / "images", arcname="images")
        tar.add(undistorted / "sparse", arcname="sparse")
    dataset_key = ctx.attempt_key("dataset.tar")
    ctx.store.put_file(dataset_key, tar_path, "application/x-tar")
    # The camera path in COLMAP's frame, recorded with the run's other artifacts.
    cameras = read_camera_poses(dataset_sparse / "images.bin")
    cameras_key = ctx.attempt_key("cameras.json")
    ctx.store.put_bytes(
        cameras_key,
        json.dumps({"frame": "colmap", "cameras": cameras}).encode(),
        "application/json",
    )
    bounds = None
    if len(points):
        lo, hi = np.percentile(points, 2, axis=0), np.percentile(points, 98, axis=0)
        bounds = {"min": lo.tolist(), "max": hi.tolist(), "frame": "colmap"}
    return finish(
        ctx,
        {"quality": quality, "sparsePoints": int(len(points)), "sparseBounds": bounds},
        outputs={"datasetKey": dataset_key, "camerasKey": cameras_key, "coverage": quality},
    )
