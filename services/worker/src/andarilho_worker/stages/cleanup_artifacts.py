"""``artifact_cleanup``: delete storage prefixes that no pipeline will complete.

The queue enqueues this job, after a delay, for two kinds of prefix:

* an attempt prefix ``tenants/{p}/runs/{r}/attempts/{stage}/{job}/{fence}/`` of an
  attempt that was lost or cancelled before it committed (the reaper's view, or a
  job cancelled before a later attempt could sweep what it left);
* a scene prefix ``tenants/{p}/scenes/{revision}/`` of a revision that is now invalid,
  because its pipeline failed or was cancelled before validation committed it.

Nothing reads what this job deletes, so it stores no manifest. Two kinds of prefix
are kept:

* the scene prefix of a revision that is valid, or that a project pointer (active
  draft or published), a scene's current draft or a share link still names. A valid
  revision is served, approved, published or shared, and a referenced one may still
  be reached through its pointer or link, so its files stay whatever its state. The
  cost: a failed layout save may still be the active draft when this job runs, and
  its files then stay until the project is deleted, because moving the pointer on
  queues no cleanup;
* the attempt prefix of a succeeded job. A succeeded camera_reconstruction holds the
  frames that layout re-texturing needs.

Every requested prefix is checked before anything is deleted.
"""

from __future__ import annotations

import logging
import re
from functools import partial
from typing import Any

from andarilho_worker.errors import StageError
from andarilho_worker.recipes import RUN_STAGES
from andarilho_worker.stages.base import StageContext, StageResult, finish

log = logging.getLogger(__name__)

_UUID = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
ATTEMPT = re.compile(
    rf"tenants/(?P<project>{_UUID})/runs/(?P<run>{_UUID})/attempts/"
    rf"(?P<stage>{'|'.join(RUN_STAGES)})/(?P<job>{_UUID})/(?P<fence>[1-9][0-9]*)/"
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
    if row["project_pointer"]:
        return "a project pointer names the revision"
    if row["scene_draft"]:
        return "the scene's current draft names the revision"
    if row["share_link"]:
        return "a share link names the revision"
    return None


def _job_keep_reason(conn: Any, *, project_id: str, job_id: str) -> str | None:
    row = conn.execute(
        "SELECT state FROM jobs WHERE id = %s AND project_id = %s", (job_id, project_id)
    ).fetchone()
    # Only a succeeded attempt holds outputs that a later stage may still read. Every
    # other state, including a job that is gone, leaves its outputs unreferenced.
    if row is not None and row["state"] == "succeeded":
        return "the attempt's job succeeded"
    return None


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
                partial(_job_keep_reason, project_id=ctx.project_id, job_id=match["job"])
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
