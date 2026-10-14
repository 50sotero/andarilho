"""``training`` (GPU): finite Splatfacto training and Gaussian splat export.

``ns-train`` runs with a fixed iteration count, the interactive viewer disabled
(``--vis tensorboard`` + ``--viewer.quit-on-train-completion True``), an explicit
output directory and timestamp, and a fixed seed. The worker then locates the
generated ``config.yml`` (never assuming a path), runs ``ns-export
gaussian-splat`` and validates the exported PLY before uploading it.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

from andarilho_worker.errors import StageError
from andarilho_worker.geometry.ply import PlyError, read_splat_ply_file
from andarilho_worker.recipes import render_argv
from andarilho_worker.runner import ProcessFailed
from andarilho_worker.stages.base import StageContext, StageResult, finish


def safe_extract(tar_path: Path, dest: Path) -> None:
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest, filter="data")  # rejects absolute paths, links outside dest


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise StageError("EXPORT_INVALID", details={"missing": pattern})
    return matches[-1]


def run(ctx: StageContext) -> StageResult:
    params = ctx.params
    work = ctx.scratch / "train"
    tar_path = work / "dataset.tar"
    ctx.store.download(ctx.output("datasetKey"), tar_path)
    data = work / "dataset"
    safe_extract(tar_path, data)
    tar_path.unlink()
    outputs = work / "outputs"
    timestamp = f"attempt-{ctx.fencing_token}"
    try:
        ctx.progress("Training the visual representation.", iterations=params.get("iterations"))
        ctx.run(
            render_argv(
                "ns_train",
                data=data,
                output_dir=outputs,
                timestamp=timestamp,
                iterations=int(params.get("iterations", 7000)),
                save_steps=int(params.get("saveSteps", 2000)),
                seed=int(params.get("seed", 42)),
            )
        )
    except ProcessFailed as exc:
        raise StageError("TRAINING_FAILED") from exc
    config = find_one(outputs, "config.yml")
    export = work / "export"
    try:
        ctx.progress("Exporting splats.")
        ctx.run(render_argv("ns_export", config=config, output_dir=export))
    except ProcessFailed as exc:
        raise StageError("EXPORT_INVALID") from exc
    ply_path = find_one(export, "*.ply")
    try:
        cloud = read_splat_ply_file(ply_path)
    except PlyError as exc:
        raise StageError("EXPORT_INVALID") from exc
    ply_key = ctx.attempt_key("splat.ply")
    ctx.store.put_file(ply_key, ply_path, "application/octet-stream")
    outputs_map = {"plyKey": ply_key, "splatCount": cloud.count}
    transforms = list(outputs.rglob("dataparser_transforms.json"))
    if transforms:
        key = ctx.attempt_key("dataparser_transforms.json")
        ctx.store.put_file(key, transforms[-1], "application/json")
        outputs_map["dataparserTransformsKey"] = key
    config_key = ctx.attempt_key("config.yml")
    ctx.store.put_file(config_key, config, "text/yaml")  # provenance only, never re-executed
    return finish(ctx, {"splats": cloud.count, "configKey": config_key}, outputs=outputs_map)
