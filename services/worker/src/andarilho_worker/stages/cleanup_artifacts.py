"""``artifact_cleanup``: delete storage prefixes that no pipeline will complete.

The queue, and the API when it cancels a run, enqueues this job, after a delay, for two
kinds of prefix:

* an attempt prefix ``tenants/{p}/runs/{r}/attempts/{stage}/{job}/{fence}/`` of an
  attempt that was lost or cancelled before it committed (the reaper's view, or a
  job cancelled before a later attempt could sweep what it left), one a worker
  failed to delete before the job ended, or the outputs of a run that ended unfinished;
* a scene prefix ``tenants/{p}/scenes/{revision}/`` of a revision that is now invalid,
  because its pipeline failed or was cancelled before validation committed it.

Nothing reads what this job deletes, so it stores no manifest. Two kinds of prefix
are kept:

* the scene prefix of a revision that is valid, or that is still pending while a
  project pointer (active draft or published), a scene's current draft or a share
  link names it: a pending revision may still be processing. An invalid revision is
  deleted whatever names it: every delivery path requires a valid revision and
  validation never revisits an invalid one, so its files are never read again. (A
  failed layout save usually stays the draft, and moving the pointer on later queues
  no cleanup, so keeping it here would keep its files until the project is deleted.)
* the attempt prefix a succeeded job committed under (its last fence), while its run
  has not failed or been cancelled. A succeeded camera_reconstruction holds the frames
  that layout re-texturing needs. The job's earlier fences are stale and deleted.

Every requested prefix is checked before anything is deleted.
"""

from __future__ import annotations

import logging
import re
from functools import partial
from typing import Any

from andarilho_worker.errors import StageError
from andarilho_worker.recipes import LEGACY_RUN_STAGES, RUN_STAGES
from andarilho_worker.stages.base import StageContext, StageResult, finish

log = logging.getLogger(__name__)

_UUID = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
# Legacy stages are accepted too: a run that ends unfinished after an upgrade queues the
# outputs of its historical jobs (e.g. training) along with the rest.
ATTEMPT = re.compile(
    rf"tenants/(?P<project>{_UUID})/runs/(?P<run>{_UUID})/attempts/"
    rf"(?P<stage>{'|'.join(RUN_STAGES + LEGACY_RUN_STAGES)})/(?P<job>{_UUID})/"
    rf"(?P<fence>[1-9][0-9]*)/"
)
SCENE = re.compile(rf"tenants/(?P<project>{_UUID})/scenes/(?P<revision>{_UUID})/")

# Every way a revision is referenced: the project's pointers, the scene's current draft
# and any share link. The composite keys make each match stay inside the project.
_REVISION = """
SELECT r.validation_state,
       EXISTS (SELECT 1 FROM projects p
               WHERE p.id = r.project_id
                 AND (p.active_draft_revision_id = r.id OR p.published_revision_id = r.id))
         AS project_pointer,
       EXISTS (SELECT 1 FROM scenes s
               WHERE s.project_id = r.project_id AND s.current_draft_revision_id = r.id)
         AS scene_draft,
       EXISTS (SELECT 1 FROM share_links l
               WHERE l.project_id = r.project_id AND l.scene_revision_id = r.id)
         AS share_link
FROM scene_revisions r
WHERE r.id = %s AND r.project_id = %s
"""


def _refuse() -> StageError:
    return StageError("INTERNAL_ERROR", "The cleanup request names a prefix outside its scope.")


def _target(project_id: str, prefix: object) -> tuple[str, str, re.Match[str]]:
    """Classify one requested prefix. Anything that is not this project's artifact is refused."""
    if isinstance(prefix, str):
        for kind, pattern in (("attempt", ATTEMPT), ("scene", SCENE)):
            match = pattern.fullmatch(prefix)
            if match is not None and match["project"] == project_id:
                return kind, prefix, match
    raise _refuse()


def _revision_keep_reason(conn: Any, *, project_id: str, revision_id: str) -> str | None:
    row = conn.execute(_REVISION, (revision_id, project_id)).fetchone()
    # No row means the project's rows are gone, so nothing can reference the objects.
    if row is None:
        return None
    if row["validation_state"] == "valid":
        return "the revision is valid"
    # Nothing serves an invalid revision, whatever names it (see the module docstring).
    if row["validation_state"] == "invalid":
        return None
    if row["project_pointer"]:
        return "a project pointer names the revision"
    if row["scene_draft"]:
        return "the scene's current draft names the revision"
    if row["share_link"]:
        return "a share link names the revision"
    return None


def _job_keep_reason(conn: Any, *, project_id: str, job_id: str, fence: int) -> str | None:
    row = conn.execute(
        "SELECT j.state, j.fencing_token, r.status AS run_status FROM jobs j "
        "LEFT JOIN reconstruction_runs r ON r.id = j.run_id AND r.project_id = j.project_id "
        "WHERE j.id = %s AND j.project_id = %s",
        (job_id, project_id),
    ).fetchone()
    # Only the attempt that succeeded holds outputs that a later stage may still read. A
    # succeeded job is never claimed or reaped again, so its fence is the one it committed
    # under. Every other attempt, including one of a job that is gone, is unreferenced.
    if row is None or row["state"] != "succeeded" or row["fencing_token"] != fence:
        return None
    # A run that ended unfinished never had a valid scene to re-texture, and a retry is
    # a new run, so nothing reads what its stages wrote.
    if row["run_status"] in ("failed", "cancelled"):
        return None
    return "the attempt's job succeeded"


def run(ctx: StageContext) -> StageResult:
    manifest = ctx.input_manifest
    prefixes = manifest.get("prefixes")
    if manifest.get("projectId") != ctx.project_id or not isinstance(prefixes, list):
        raise _refuse()
    # Validate every prefix before deleting any, so a bad entry cannot part-complete a request.
    targets = [_target(ctx.project_id, prefix) for prefix in prefixes]
    deleted: list[str] = []
    kept: list[str] = []
    for kind, prefix, match in targets:
        if kind == "scene":
            reason = ctx.query(
                partial(
                    _revision_keep_reason,
                    project_id=ctx.project_id,
                    revision_id=match["revision"],
                )
            )
        else:
            reason = ctx.query(
                partial(
                    _job_keep_reason,
                    project_id=ctx.project_id,
                    job_id=match["job"],
                    fence=int(match["fence"]),
                )
            )
        if reason is not None:
            log.info("artifact cleanup keeps %s: %s", prefix, reason)
            kept.append(prefix)
            continue
        ctx.store.delete_prefix(prefix)
        ctx.store.abort_multipart_uploads(prefix)
        deleted.append(prefix)
    return finish(
        ctx,
        {"deleted": deleted, "kept": kept, "sceneRevisionId": manifest.get("sceneRevisionId")},
        store_manifest=False,
    )
