"""Which revision a project is published on, derived from its live share links.

``projects.published_revision_id`` is updated when links are created or revoked,
but links also expire on their own. Reads therefore derive publication from the
links that are still live: the stored pointer while it has one, otherwise the
newest live link's revision, otherwise nothing.

An expiry changes the publication without any request, so nothing bumps the
versions of the scenes involved when it happens. ``settle_expired_publication``
does that for the first request that touches the project's scenes afterwards.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from andarilho_api.deps import utcnow
from andarilho_api.models import Project, Scene, SceneRevision, ShareLink


def live_published(db: Session, projects: Iterable[Project]) -> dict[uuid.UUID, uuid.UUID | None]:
    projects = list(projects)
    result: dict[uuid.UUID, uuid.UUID | None] = {p.id: None for p in projects}
    candidates = [p.id for p in projects if p.published_revision_id is not None]
    if not candidates:
        return result
    rows = db.execute(
        select(ShareLink.project_id, ShareLink.scene_revision_id)
        .where(
            ShareLink.project_id.in_(candidates),
            ShareLink.revoked_at.is_(None),
            or_(ShareLink.expires_at.is_(None), ShareLink.expires_at > utcnow()),
        )
        .order_by(ShareLink.created_at.desc(), ShareLink.id)
    ).all()
    live: dict[uuid.UUID, list[uuid.UUID]] = {}
    for project_id, revision_id in rows:
        live.setdefault(project_id, []).append(revision_id)
    for p in projects:
        revisions = live.get(p.id, [])
        if p.published_revision_id in revisions:
            result[p.id] = p.published_revision_id
        elif revisions:
            result[p.id] = revisions[0]
    return result


def published_revision_id(db: Session, project: Project) -> uuid.UUID | None:
    return live_published(db, [project])[project.id]


def publication_changed(
    db: Session, revision_ids: Iterable[uuid.UUID | None], *, skip: uuid.UUID | None = None
) -> None:
    """Bump the version of each scene holding one of these revisions (except ``skip``).

    Publishing is guarded by the scene ETag: a tab that read a scene before its
    revision was published or unpublished must get a conflict, not publish over it.
    """
    ids = {r for r in revision_ids if r is not None}
    if not ids:
        return
    holders = select(SceneRevision.scene_id).where(SceneRevision.id.in_(ids))
    for held in db.execute(
        select(Scene).where(Scene.id.in_(holders), Scene.id != skip).with_for_update()
    ).scalars():
        held.version += 1


def settle_expired_publication(db: Session, project: Project) -> bool:
    """Catch the stored pointer up with links that expired since it was last set.

    When it differs from the live-derived publication, the pointer takes that value and
    the scenes holding either revision get a new version, as an explicit revoke would
    give them. The caller holds the project's row lock and commits. Returns whether
    anything changed.
    """
    effective = published_revision_id(db, project)
    stored = project.published_revision_id
    if effective == stored:
        return False
    project.published_revision_id = effective
    project.updated_at = utcnow()
    publication_changed(db, [stored, effective])
    return True
