"""Project membership and role checks.

Every resource lookup joins ``project_members`` on the caller's verified subject.
A resource that does not exist and one the caller cannot see both produce the
same ``404 NOT_FOUND``. ``403 FORBIDDEN`` is reserved for members whose role is
insufficient.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from andarilho_api.errors import forbidden, not_found, project_deleting
from andarilho_api.models import Project, ProjectMember

OWNER = ("owner",)
WRITERS = ("owner", "editor")
ANY_ROLE = ("owner", "editor", "viewer")


def load_project(
    db: Session,
    project_id: uuid.UUID,
    subject: str,
    *,
    roles: tuple[str, ...] = ANY_ROLE,
    lock: bool = False,
    require_active: bool = False,
) -> tuple[Project, str]:
    """Return ``(project, role)`` for a member. ``lock`` takes ``FOR UPDATE`` on the
    project row, serializing the mutation against deletion and other writers."""
    stmt = (
        select(Project, ProjectMember.role)
        .join(
            ProjectMember,
            and_(ProjectMember.project_id == Project.id, ProjectMember.subject == subject),
        )
        .where(Project.id == project_id, Project.lifecycle != "deleted")
    )
    if lock:
        stmt = stmt.with_for_update(of=Project).execution_options(populate_existing=True)
    row = db.execute(stmt).one_or_none()
    if row is None:
        raise not_found("project")
    project, role = row
    if role not in roles:
        raise forbidden()
    if require_active and project.lifecycle != "active":
        raise project_deleting()
    return project, role


def load_scoped[T](
    db: Session, model: type[T], resource_id: uuid.UUID, subject: str, what: str
) -> T:
    """Load a project-scoped row only if the caller is a member of its project."""
    column: Any = model.project_id  # type: ignore[attr-defined]
    stmt = (
        select(model)
        .join(Project, Project.id == column)
        .join(
            ProjectMember,
            and_(ProjectMember.project_id == column, ProjectMember.subject == subject),
        )
        .where(model.id == resource_id, Project.lifecycle != "deleted")  # type: ignore[attr-defined]
    )
    obj = db.execute(stmt).scalar_one_or_none()
    if obj is None:
        raise not_found(what)
    return obj


def lock_row[T](db: Session, model: type[T], resource_id: uuid.UUID) -> T:
    stmt = (
        select(model)
        .where(model.id == resource_id)  # type: ignore[attr-defined]
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    return db.execute(stmt).scalar_one()
