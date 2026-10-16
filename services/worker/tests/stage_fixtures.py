"""A stage context for tests that call a stage function directly, with no supervisor."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from andarilho_worker.recipes import recipe_for
from andarilho_worker.stages.base import CancellationToken, JobInfo, StageContext, StageServices


def stage_context(
    tmp_path: Path,
    store: Any,
    settings: Any,
    outputs: dict[str, Any],
    *,
    stage: str = "normalizing",
    project_id: str = "p",
    run_id: str = "run",
    quality: str = "preview",
) -> StageContext:
    """A context whose job has ``outputs`` from earlier stages. The runner is left out:
    the stages under test run no tools.
    """
    recipe = recipe_for(quality)
    deadline = datetime.now(UTC) + timedelta(hours=1)
    return StageContext(
        job_id="job",
        run_id=run_id,
        stage=stage,
        attempt=1,
        fencing_token=1,
        scratch=tmp_path / "scratch",
        recipe_digest=recipe.digest,
        recipe=recipe.as_dict(),
        input_manifest={"outputs": outputs},
        deadline=deadline,
        max_scratch_bytes=settings.max_scratch_bytes,
        cancellation=CancellationToken(deadline),
        job=JobInfo(project_id, run_id, None, None, "cpu"),
        services=StageServices(store=store, runner=None, settings=settings),
    )
