"""Run the real capture pipeline on one local capture, without a database.

Used by the manual GPU smoke workflow and for local experiments:

    python -m andarilho_worker.smoke --video /data/capture.mp4 --out /out [--quality preview]
    python -m andarilho_worker.smoke --dataset /data/dataset.tar --out /out [--layout l.json]
    ... [--analysis | --analysis-reply reply.json]

``--video`` drives ``preparing`` → ``camera_reconstruction`` → ``normalizing`` →
``scene_analysis`` (a GPU host) through the same stage functions, pinned recipe,
subprocess supervision and output contracts as the worker, using a local object store.
``--dataset`` starts from the ``dataset.tar`` a reconstruction stored (undistorted
frames plus the COLMAP model) and runs only ``normalizing`` and ``scene_analysis``, so
it needs no GPU.

The scene analysis is off unless asked for. ``--analysis`` calls the model that the
``AI_*`` and ``OPENCODE_API_KEY_FILE`` variables configure, as the worker does.
``--analysis-reply`` answers with a recorded reply instead (the model's JSON: rooms,
frames and objects), so a run can be reproduced, or checked with a hand-written reply,
offline. Either way ``--out/analysis`` receives what the model is shown, each image as
``frame-<n>.jpg``, and the reply.

In the worker the proposal, texturing and packaging stages read the database; here
the same functions run without it. ``--out/bundle`` receives what the product would
publish: the automatic proposal with the rooms, doors and furniture of the analysis
(or the ``--layout`` file, in the run's canonical world, standing in for an edited
layout), the model textured from the frames, its collision GLB, the poster and a scene
manifest, checked by the same scene validator (the viewer's dev page opens it with
``?bundle=``). A JSON report is written to ``--out``; the process exits non-zero if
any stage fails or the bundle does not validate, so a broken toolchain cannot pass
silently.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import tarfile
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

from andarilho_worker.ai import Completion, VisionClient
from andarilho_worker.artifacts import LocalObjectStore, ObjectStore
from andarilho_worker.errors import StageError
from andarilho_worker.geometry.layout import collision_glb, validate_layout
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
from andarilho_worker.stages.package_scene import GEOMETRY_SOURCE, unseen_warning
from andarilho_worker.stages.propose_geometry import proposed_layout
from andarilho_worker.stages.texture import paint
from andarilho_worker.validation import canonical_json, validate_scene

CAPTURE_STAGES = ("preparing", "camera_reconstruction", "normalizing", "scene_analysis")
DATASET_STAGES = ("normalizing", "scene_analysis")
DATASET_WARNING = "Started from a stored dataset: the frame selection was not rerun."
# The worker's AI variables, the only part of the environment the smoke runner reads.
AI_VARIABLES = re.compile(r"AI_[A-Z_]+|OPENCODE_API_KEY_FILE")
IMAGE_LABEL = re.compile(r"Frame (\d+)( \(photo\))?:|Floor plan:")


class AnalysisRecorder:
    """The scene analysis's model client: it records the request and the reply.

    Each image the stage sends is saved under ``folder``, named after the label before
    it (``frame-<n>.jpg``, ``frame-<n>-photo.jpg`` or ``floor-plan.jpg``), with the text
    in ``request.txt``. With ``reply`` it answers with that recorded reply instead of
    calling ``client``; the reply is written to ``reply.json`` either way.
    """

    def __init__(self, folder: Path, client: Any = None, reply: dict[str, Any] | None = None):
        if (client is None) == (reply is None):
            raise ValueError("exactly one of client and reply is required")
        self.folder, self.client, self.reply = folder, client, reply

    def complete_json(
        self, system: str, content: list[dict[str, Any]], *, session: str | None = None
    ) -> Completion:
        self.folder.mkdir(parents=True, exist_ok=True)
        texts, name = [system], "image"
        for part in content:
            if part["type"] == "text":
                texts.append(part["text"])
                label = IMAGE_LABEL.fullmatch(part["text"])
                if label is None:
                    name = "image"
                elif label.group(1) is None:
                    name = "floor-plan"
                else:
                    name = f"frame-{label.group(1)}" + ("-photo" if label.group(2) else "")
            else:
                data = part["image_url"]["url"].split(",", 1)[1]
                (self.folder / f"{name}.jpg").write_bytes(base64.b64decode(data))
                texts.append(f"[{name}.jpg]")
        (self.folder / "request.txt").write_text("\n\n".join(texts), encoding="utf-8")
        if self.reply is not None:
            completion = Completion(data=self.reply, model="recorded", usage={})
        else:
            completion = self.client.complete_json(system, content, session=session)
        reply = json.dumps(completion.data, indent=2)
        (self.folder / "reply.json").write_text(reply, encoding="utf-8")
        return completion


def _settings(work: Path, capability: str, analysis: bool) -> Settings:
    """Settings for a local run: no database or object store, and the worker's AI variables."""
    env = {k: v for k, v in os.environ.items() if AI_VARIABLES.fullmatch(k)}
    env |= {"DATABASE_URL": "postgresql://unused", "AI_SCENE_ANALYSIS": str(analysis).lower()}
    settings = Settings.from_env(env)
    return replace(settings, worker_capability=capability, worker_scratch_dir=work)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _context(base: StageContext, stage: str, outputs: Mapping[str, Any]) -> StageContext:
    scratch = base.scratch / stage
    scratch.mkdir(parents=True, exist_ok=True)
    return replace(base, stage=stage, scratch=scratch, input_manifest={"outputs": dict(outputs)})


def _video_inputs(store: ObjectStore, video: Path, project_id: str) -> dict[str, Any]:
    upload_id = str(uuid.uuid4())
    key = f"tenants/{project_id}/originals/{upload_id}"
    store.put_file(key, video, "video/mp4")
    return {
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
        ]
    }


def _dataset_outputs(base: StageContext, dataset: Path) -> dict[str, Any]:
    """What ``camera_reconstruction`` would have output for a stored ``dataset.tar``.

    The archive holds only the registered frames, so their count stands in for both
    the registered and the selected views.
    """
    with tarfile.open(dataset) as tar:
        frames = sum(1 for m in tar.getmembers() if m.isfile() and m.name.startswith("images/"))
    key = replace(base, stage="camera_reconstruction").attempt_key("dataset.tar")
    base.store.put_file(key, dataset, "application/x-tar")
    coverage = {
        "registeredViews": frames,
        "selectedViews": frames,
        "disconnectedComponents": 1,
        "registeredRatio": 1.0,
        "warnings": [DATASET_WARNING],
    }
    return {"datasetKey": key, "coverage": coverage}


def write_bundle(
    store: ObjectStore,
    outputs: Mapping[str, Any],
    layout: dict[str, Any],
    layout_source: str,
    bundle: Path,
    *,
    project_id: str,
    run_id: str,
    recipe_digest: str,
) -> dict[str, Any]:
    """Package the textured scene as ``packaging`` would, then run the scene validator.

    Like every packaged scene, the bundle is walkable as it is (``approved_collision``).
    """
    bundle.mkdir(parents=True, exist_ok=True)
    revision_id = str(uuid.uuid4())
    prefix = scene_prefix(project_id, revision_id)
    model_path = bundle / "model.glb"
    model_sha = store.download(str(outputs["modelKey"]), model_path)
    collision_path = bundle / "collision.glb"
    collision_path.write_bytes(collision_glb(layout))
    files = {prefix + model_path.name: model_path, prefix + collision_path.name: collision_path}
    poster = None
    if outputs.get("posterKey"):
        poster_path = bundle / "poster.webp"
        store.download(str(outputs["posterKey"]), poster_path)
        files[prefix + poster_path.name] = poster_path
        poster = asset_entry(prefix + poster_path.name, poster_path, "webp")
    quality = {k: v for k, v in dict(outputs["coverage"]).items() if k != "registeredRatio"}
    warnings = [
        *quality.get("warnings", []),
        *(outputs.get("frameWarnings") or []),
        *(outputs.get("analysisWarnings") or []),
    ]
    unseen = unseen_warning(outputs["textureCoverage"])
    quality["warnings"] = warnings if unseen is None else [*warnings, unseen]
    scale_status = "user_measured" if layout.get("calibration") else str(outputs["scaleStatus"])
    manifest = build_manifest(
        scene_id=str(uuid.uuid4()),
        scene_revision_id=revision_id,
        revision=1,
        run_id=run_id,
        layout=layout,
        layout_revision=1,
        render=asset_entry(prefix + model_path.name, model_path, "glb", sha256=model_sha),
        collision=asset_entry(prefix + collision_path.name, collision_path, "glb"),
        poster=poster,
        scale_status=scale_status,
        geometry_source=GEOMETRY_SOURCE[layout_source],
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
        "renderFile": f"{bundle.name}/{model_path.name}",
        "renderBytes": model_path.stat().st_size,
        "renderSha256": model_sha,
        "posterFile": f"{bundle.name}/poster.webp" if poster else None,
        "bounds": manifest["bounds"],
        "spawn": manifest["spawn"],
        "navigation": manifest["navigation"]["mode"],
        "findings": findings,
    }


def run_smoke(
    out: Path,
    quality: str,
    max_seconds: int,
    *,
    video: Path | None = None,
    dataset: Path | None = None,
    layout_file: Path | None = None,
    analysis: bool = False,
    analysis_reply: Path | None = None,
) -> dict[str, Any]:
    """Run the stages, then propose, texture and package; see the module docstring.

    ``analysis`` calls the configured model in ``scene_analysis``, and ``analysis_reply``
    answers with that recorded reply instead. With neither, the stage is switched off.
    """
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    store = LocalObjectStore(work / "store")
    project_id, run_id = str(uuid.uuid4()), str(uuid.uuid4())
    recipe = recipe_for(quality)
    capability = "gpu" if video is not None else "cpu"
    settings = _settings(work, capability, analysis or analysis_reply is not None)
    vision = None
    if analysis_reply is not None:
        reply = json.loads(analysis_reply.read_text(encoding="utf-8"))
        vision = AnalysisRecorder(out / "analysis", reply=reply)
    elif analysis:
        vision = AnalysisRecorder(out / "analysis", client=VisionClient.from_settings(settings))
    services = StageServices(
        store=store,
        runner=ProcessRunner(),
        settings=settings,
        progress=lambda message, data: print(f"  {message}", flush=True) or True,
        vision=vision,
    )
    deadline = datetime.now(UTC) + timedelta(seconds=max_seconds)
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
        job=JobInfo(project_id, run_id, None, None, settings.worker_capability),
        services=services,
    )
    outputs: dict[str, Any] = {"captureSetId": str(uuid.uuid4()), "quality": quality}
    stage_outputs: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {"quality": quality, "recipeDigest": recipe.digest, "stages": []}

    def timed(stage: str, step: Callable[[], Any]) -> Any:
        print(f"[{stage}]", flush=True)
        started = time.monotonic()
        result = step()
        report["stages"].append({"stage": stage, "seconds": round(time.monotonic() - started, 1)})
        return result

    if video is not None and dataset is None:
        outputs |= _video_inputs(store, video, project_id)
        stages = CAPTURE_STAGES
    elif dataset is not None and video is None:
        stage_outputs["camera_reconstruction"] = _dataset_outputs(base, dataset)
        outputs |= stage_outputs["camera_reconstruction"]
        stages = DATASET_STAGES
    else:
        raise ValueError("exactly one of video and dataset is required")
    for stage in stages:
        result = timed(stage, partial(STAGES[stage], _context(base, stage, outputs)))
        stage_outputs[stage] = dict(result.outputs)
        outputs |= stage_outputs[stage]
        if "frame" in result.manifest:
            report["frame"] = result.manifest["frame"]
        if stage == "scene_analysis":
            warnings = outputs.get("analysisWarnings") or []
            report["analysis"] = {**result.manifest, "warnings": warnings}

    diagnostics: dict[str, Any] = {}
    if layout_file is not None:
        layout, layout_source = json.loads(layout_file.read_text(encoding="utf-8")), "user"
    else:
        layout, diagnostics = proposed_layout(_context(base, "geometry_proposal", outputs))
        layout_source = "proposal"
    findings = [f.as_dict() for f in validate_layout(layout)]
    report.update(layoutSource=layout_source, proposal=diagnostics)
    if findings:
        report["bundle"] = {"findings": findings}
    else:
        textured = timed(
            "texturing", lambda: paint(_context(base, "texturing", outputs), layout, stage_outputs)
        )
        outputs |= dict(textured.outputs)
        report["texturing"] = {
            k: textured.manifest[k] for k in ("frames", "texelSize", "pages", "modelBytes")
        }
        report["bundle"] = write_bundle(
            store,
            outputs,
            layout,
            layout_source,
            out / "bundle",
            project_id=project_id,
            run_id=run_id,
            recipe_digest=recipe.digest,
        )
    report.update(
        coverage=outputs.get("coverage"),
        textureCoverage=outputs.get("textureCoverage"),
        frameCount=outputs.get("frameCount"),
        worldFromColmap=outputs.get("worldFromColmap"),
    )
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layout", type=Path)
    ai = parser.add_mutually_exclusive_group()
    ai.add_argument("--analysis", action="store_true", help="call the configured AI model")
    ai.add_argument("--analysis-reply", type=Path, help="answer with this recorded reply")
    parser.add_argument("--quality", choices=("preview", "standard"), default="preview")
    parser.add_argument("--max-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    try:
        report = run_smoke(
            args.out,
            args.quality,
            args.max_seconds,
            video=args.video,
            dataset=args.dataset,
            layout_file=args.layout,
            analysis=args.analysis,
            analysis_reply=args.analysis_reply,
        )
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
