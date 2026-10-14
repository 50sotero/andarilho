"""The training stage: finite training, export with its own environment, PLY upload."""

from __future__ import annotations

import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from andarilho_worker.geometry.ply import write_splat_ply
from andarilho_worker.geometry.synthetic import synthetic_splats
from andarilho_worker.recipes import recipe_for
from andarilho_worker.runner import RunResult, minimal_env
from andarilho_worker.stages import train
from andarilho_worker.stages.base import (
    CancellationToken,
    JobInfo,
    StageContext,
    StageServices,
)
from andarilho_worker.stages.normalize import synthetic_layout

PLY = write_splat_ply(synthetic_splats(synthetic_layout()))


class FakeTrainer:
    """Stands in for ns-train/ns-export: records argv and env, writes their outputs."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, str] | None]] = []

    def run(self, argv, *, timeout, check_cancelled=None, env=None, **kw):
        argv = [str(a) for a in argv]
        self.calls.append((argv, dict(env) if env is not None else None))
        out = Path(argv[argv.index("--output-dir") + 1])
        if argv[0] == "ns-train":
            run_dir = out / "andarilho" / "splatfacto" / argv[argv.index("--timestamp") + 1]
            run_dir.mkdir(parents=True)
            (run_dir / "config.yml").write_text("method_name: splatfacto\n")
            (run_dir.parent.parent / "dataparser_transforms.json").write_text("{}")
        elif argv[0] == "ns-export":
            out.mkdir(parents=True, exist_ok=True)
            (out / "splat.ply").write_bytes(PLY)
        return RunResult(tuple(argv), 0, "", "", 0.01)


def _context(tmp_path, store, settings, runner) -> StageContext:
    dataset = tmp_path / "dataset.tar"
    with tarfile.open(dataset, "w") as tar:
        images = tmp_path / "images"
        images.mkdir()
        (images / "frame_00001.jpg").write_bytes(b"\xff\xd8\xff")
        tar.add(images, arcname="images")
    store.put_file("tenants/p/runs/r/dataset.tar", dataset, "application/x-tar")
    recipe = recipe_for("preview")
    deadline = datetime.now(UTC) + timedelta(hours=1)
    return StageContext(
        job_id="job",
        run_id="r",
        stage="training",
        attempt=1,
        fencing_token=1,
        scratch=tmp_path / "scratch",
        recipe_digest=recipe.digest,
        recipe=recipe.as_dict(),
        input_manifest={"outputs": {"datasetKey": "tenants/p/runs/r/dataset.tar"}},
        deadline=deadline,
        max_scratch_bytes=settings.max_scratch_bytes,
        cancellation=CancellationToken(deadline),
        job=JobInfo("p", "r", None, None, "gpu"),
        services=StageServices(store=store, runner=runner, settings=settings,
                               progress=lambda message, data: True),  # fmt: skip
    )


def test_training_exports_with_the_full_unpickler_for_its_own_checkpoint(
    tmp_path, store, settings, monkeypatch
):
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", "/opt/torch_extensions")
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret")
    trainer = FakeTrainer()
    result = train.run(_context(tmp_path, store, settings, trainer))
    (train_argv, train_env), (export_argv, export_env) = trainer.calls
    assert train_argv[:2] == ["ns-train", "splatfacto"]
    assert train_env is None  # the runner's minimal environment
    assert export_argv[:2] == ["ns-export", "gaussian-splat"]
    assert export_env == minimal_env({"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"})
    assert export_env["TORCH_EXTENSIONS_DIR"] == "/opt/torch_extensions"
    assert "DATABASE_URL" not in export_env
    assert result.outputs["splatCount"] > 0
    assert store.get_bytes(result.outputs["plyKey"]) == PLY
    assert "dataparserTransformsKey" in result.outputs


def test_subprocesses_get_the_precompiled_kernel_location_and_no_secrets(monkeypatch):
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", "/opt/torch_extensions")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9;9.0;12.0")
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "do-not-pass")
    env = minimal_env()
    assert env["TORCH_EXTENSIONS_DIR"] == "/opt/torch_extensions"
    assert env["TORCH_CUDA_ARCH_LIST"] == "8.0;8.6;8.9;9.0;12.0"
    assert "S3_SECRET_ACCESS_KEY" not in env
