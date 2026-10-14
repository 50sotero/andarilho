"""Run the real capture pipeline on one local video, without a database.

Used by the manual GPU smoke workflow and for local experiments on a GPU host:

    python -m andarilho_worker.smoke --video /data/capture.mp4 --out /out [--quality preview]

It drives ``preparing`` → ``camera_reconstruction`` → ``training`` →
``normalizing`` through the same stage functions, pinned recipe, subprocess
supervision and output contracts as the worker, using a local object store.
The geometry and packaging stages need the database and are not run. The
render asset and a JSON report are written to ``--out``; the process exits
non-zero if any stage fails, so a broken toolchain cannot pass silently.
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

from andarilho_worker.artifacts import LocalObjectStore
from andarilho_worker.errors import StageError
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

CAPTURE_STAGES = ("preparing", "camera_reconstruction", "training", "normalizing")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    render = store.get_bytes(outputs["renderKey"], max_bytes=4 * 1024**3)
    target = out / f"scene.{outputs['renderFormat']}"
    target.write_bytes(render)
    report.update(
        renderFile=target.name,
        renderBytes=len(render),
        renderSha256=hashlib.sha256(render).hexdigest(),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
