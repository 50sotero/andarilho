"""Run the real capture pipeline on one local video, without a database.

Used by the manual GPU smoke workflow and for local experiments on a GPU host:

    python -m andarilho_worker.smoke --video /data/capture.mp4 --out /out [--quality preview]

It drives ``preparing`` → ``camera_reconstruction`` → ``training`` →
``normalizing`` through the same stage functions, pinned recipe, subprocess
supervision and output contracts as the worker, using a local object store.
The geometry and packaging stages need the database and are not run; instead
``--out/bundle`` receives what they would package for review, built by the same
functions: the render asset, the automatic single-room layout proposal, its
collision GLB, the poster and a scene manifest, checked by the same scene validator (the
viewer's dev page opens it with ``?bundle=``). A JSON report is written to
``--out``; the process exits non-zero if any stage fails or the bundle does not
validate, so a broken toolchain cannot pass silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from andarilho_worker.artifacts import LocalObjectStore, ObjectStore
from andarilho_worker.errors import StageError
from andarilho_worker.geometry.layout import collision_glb, rectangle_layout, validate_layout
from andarilho_worker.manifest import asset_entry, build_manifest, scene_prefix
from andarilho_worker.recipes import recipe_for
from andarilho_worker.runner import ProcessRunner
from andarilho_worker.settings import Settings
from andarilho_worker.stages import STAGES
from andarilho_worker.stages.base import (
    CancellationToken,
    JobInfo,
    StageContext,
    StageServices,
)
from andarilho_worker.validation import canonical_json, validate_scene

CAPTURE_STAGES = ("preparing", "camera_reconstruction", "training", "normalizing")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bundle(
    store: ObjectStore,
    outputs: dict[str, Any],
    bundle: Path,
    *,
    project_id: str,
    run_id: str,
    recipe_digest: str,
) -> dict[str, Any]:
    """Package the normalized scene as review would see it, without a database.

    Mirrors ``geometry_proposal`` (rectangle layout from the proposal bounds) and
    ``packaging`` (manifest), then runs the scene validator. The proposal stays
    ``inspection_only``: walking it needs a reviewed layout, as in the product.
    """
    bounds = outputs["proposalBounds"]
    layout = rectangle_layout(
        bounds["minXZ"], bounds["maxXZ"], elevation=bounds["floorY"], height=bounds["height"]
    )
    findings = [f.as_dict() for f in validate_layout(layout)]
    if findings:
        return {"findings": findings}
    bundle.mkdir(parents=True, exist_ok=True)
    revision_id = str(uuid.uuid4())
    prefix = scene_prefix(project_id, revision_id)
    render_format = str(outputs["renderFormat"])
    render_path = bundle / f"scene.{render_format}"
    render_sha = store.download(str(outputs["renderKey"]), render_path)
    collision_path = bundle / "collision.glb"
    collision_path.write_bytes(collision_glb(layout))
    files = {prefix + render_path.name: render_path, prefix + collision_path.name: collision_path}
    poster = None
    if outputs.get("posterKey"):
        poster_path = bundle / "poster.webp"
        store.download(str(outputs["posterKey"]), poster_path)
        files[prefix + poster_path.name] = poster_path
        poster = asset_entry(prefix + poster_path.name, poster_path, "webp")
    quality = {k: v for k, v in (outputs.get("coverage") or {}).items() if k != "registeredRatio"}
    manifest = build_manifest(
        scene_id=str(uuid.uuid4()),
        scene_revision_id=revision_id,
        revision=1,
        run_id=run_id,
        layout=layout,
        layout_revision=1,
        render=asset_entry(
            prefix + render_path.name, render_path, render_format, sha256=render_sha
        ),
        collision=asset_entry(prefix + collision_path.name, collision_path, "glb"),
        poster=poster,
        world_from_decoded_render=outputs["worldFromDecodedRender"],
        scale_status=str(outputs["scaleStatus"]),
        geometry_source="proposal",
        recipe_digest=recipe_digest,
        capture_set_id=str(outputs["captureSetId"]),
        quality=quality,
    )

    def fetch(key: str) -> Path:
        if key not in files:
            raise FileNotFoundError(key)
        return files[key]

    findings = [
        f.as_dict() for f in validate_scene(manifest, fetch, project_id=project_id, layout=layout)
    ]
    (bundle / "manifest.json").write_bytes(canonical_json(manifest))
    (bundle / "layout.json").write_text(json.dumps(layout, indent=2), encoding="utf-8")
    return {
        "renderFile": f"{bundle.name}/{render_path.name}",
        "renderBytes": render_path.stat().st_size,
        "renderSha256": render_sha,
        "posterFile": f"{bundle.name}/poster.webp" if poster else None,
        "bounds": manifest["bounds"],
        "spawn": manifest["spawn"],
        "findings": findings,
    }


def run_smoke(video: Path, out: Path, quality: str, max_seconds: int) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    store = LocalObjectStore(work / "store")
    project_id, run_id, upload_id = (str(uuid.uuid4()) for _ in range(3))
    key = f"tenants/{project_id}/originals/{upload_id}"
    store.put_file(key, video, "video/mp4")
    recipe = recipe_for(quality)
    settings = Settings(
        database_url="postgresql://unused", worker_capability="gpu", worker_scratch_dir=work
    )
    services = StageServices(
        store=store,
        runner=ProcessRunner(),
        settings=settings,
        progress=lambda message, data: print(f"  {message}", flush=True) or True,
    )
    outputs: dict[str, Any] = {
        "captureSetId": str(uuid.uuid4()),
        "quality": quality,
        "inputs": [
            {
                "uploadId": upload_id,
                "mediaKind": "video",
                "key": key,
                "storageVersion": None,
                "sha256": _sha256(video),
                "bytes": video.stat().st_size,
                "contentType": "video/mp4",
                "metadata": {},
            }
        ],
    }
    deadline = datetime.now(UTC) + timedelta(seconds=max_seconds)
    report: dict[str, Any] = {"quality": quality, "recipeDigest": recipe.digest, "stages": []}
    base = StageContext(
        job_id="smoke",
        run_id=run_id,
        stage="",
        attempt=1,
        fencing_token=1,
        scratch=work,
        recipe_digest=recipe.digest,
        recipe=recipe.as_dict(),
        input_manifest={},
        deadline=deadline,
        max_scratch_bytes=settings.max_scratch_bytes,
        cancellation=CancellationToken(deadline),
        job=JobInfo(project_id, run_id, None, None, "gpu"),
        services=services,
    )
    for stage in CAPTURE_STAGES:
        print(f"[{stage}]", flush=True)
        scratch = work / stage
        scratch.mkdir(parents=True, exist_ok=True)
        ctx = replace(base, stage=stage, scratch=scratch, input_manifest={"outputs": outputs})
        started = time.monotonic()
        result = STAGES[stage](ctx)
        outputs = {**outputs, **dict(result.outputs)}
        report["stages"].append({"stage": stage, "seconds": round(time.monotonic() - started, 1)})
    bundle = write_bundle(
        store,
        outputs,
        out / "bundle",
        project_id=project_id,
        run_id=run_id,
        recipe_digest=recipe.digest,
    )
    report.update(
        bundle=bundle,
        coverage=outputs.get("coverage"),
        splatCount=outputs.get("splatCount"),
        frameCount=outputs.get("frameCount"),
        worldFromDecodedRender=outputs.get("worldFromDecodedRender"),
    )
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quality", choices=("preview", "standard"), default="preview")
    parser.add_argument("--max-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    try:
        report = run_smoke(args.video, args.out, args.quality, args.max_seconds)
    except StageError as exc:
        print(f"smoke failed: {exc.code}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    if report["bundle"]["findings"]:
        print("smoke failed: the packaged scene does not validate", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
