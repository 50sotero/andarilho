"""Shared FastAPI dependencies and helpers."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from fastapi import Request

from andarilho_api.errors import precondition_required, version_conflict
from andarilho_api.settings import Settings
from andarilho_api.storage import Storage

_IF_MATCH = re.compile(r'^\s*(?:W/)?"?(\d{1,10})"?\s*$')


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_storage(request: Request) -> Storage:
    return request.app.state.storage


def utcnow() -> datetime:
    return datetime.now(UTC)


def check_if_match(request: Request, current_version: int, what: str) -> None:
    """Missing ``If-Match`` -> 428; stale or malformed -> 409 VERSION_CONFLICT."""
    value = request.headers.get("if-match")
    if value is None or not value.strip():
        raise precondition_required("If-Match")
    match = _IF_MATCH.match(value)
    if match is None or int(match.group(1)) != current_version:
        raise version_conflict(what)
