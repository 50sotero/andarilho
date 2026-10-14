"""Structured error envelope, request IDs and exception handlers.

Every error response has the shape documented in docs/api-contract.md::

    {"error": {"code", "message", "retryable", "suggestedAction", "requestId"}}

Messages are fixed, safe strings. Raw input, SQL, filesystem paths, subprocess
output and signed URLs never reach a client.
"""

from __future__ import annotations

import logging
import re
import secrets
from contextvars import ContextVar
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("andarilho_api")

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_SAFE_LOC = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        suggested_action: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retryable = retryable
        self.suggested_action = suggested_action
        self.headers = headers or {}


def not_found(what: str = "resource") -> ApiError:
    return ApiError(404, "NOT_FOUND", f"The {what} was not found.")


def forbidden() -> ApiError:
    return ApiError(403, "FORBIDDEN", "Your role on this project does not allow this action.")


def unauthenticated(message: str = "Authentication is required.") -> ApiError:
    return ApiError(
        401,
        "UNAUTHENTICATED",
        message,
        suggested_action="Sign in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def validation_failed(message: str) -> ApiError:
    return ApiError(422, "VALIDATION_FAILED", message)


def invalid_state(message: str, suggested_action: str | None = None) -> ApiError:
    return ApiError(409, "INVALID_STATE", message, suggested_action=suggested_action)


def project_deleting() -> ApiError:
    return ApiError(409, "PROJECT_DELETING", "This project is being deleted.")


def precondition_required(header: str) -> ApiError:
    return ApiError(
        428,
        "PRECONDITION_REQUIRED",
        f"The {header} header is required for this request.",
        suggested_action=f"Send the {header} header and try again.",
    )


def version_conflict(what: str = "resource") -> ApiError:
    return ApiError(
        409,
        "VERSION_CONFLICT",
        f"The {what} changed since you loaded it.",
        suggested_action=f"Reload the {what} and apply your change again.",
    )


def current_request_id() -> str:
    return request_id_var.get()


def error_body(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    suggested_action: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "suggestedAction": suggested_action,
            "requestId": request_id or current_request_id(),
        }
    }


def _request_id_from(request: Request) -> str:
    return getattr(request.state, "request_id", None) or current_request_id()


def _response(request: Request, status: int, body: dict[str, Any], headers=None) -> JSONResponse:
    rid = _request_id_from(request)
    body["error"]["requestId"] = rid
    merged = {"X-Request-ID": rid, "Cache-Control": "private, no-store"}
    merged.update(headers or {})
    return JSONResponse(status_code=status, content=body, headers=merged)


def _safe_loc(loc: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for item in loc:
        if isinstance(item, int):
            parts.append(str(item))
        elif isinstance(item, str) and _SAFE_LOC.match(item):
            parts.append(item)
        else:
            parts.append("<field>")
    return ".".join(parts) or "request"


def _validation_message(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "The request is invalid."
    first = errors[0]
    kind = first.get("type", "invalid")
    loc = _safe_loc(tuple(first.get("loc", ())))
    if kind == "extra_forbidden":
        detail = "unknown field"
    elif kind in ("json_invalid", "json_type"):
        detail = "malformed JSON body"
    elif kind == "missing":
        detail = "field required"
    else:
        # Pydantic messages describe the constraint, not the submitted value. Our own
        # contract validators only mention identifiers that already matched a safe
        # pattern. Strip anything after a newline and bound the length.
        detail = str(first.get("msg", "invalid value")).splitlines()[0][:200]
        detail = detail.removeprefix("Value error, ")
    more = f" ({len(errors) - 1} more)" if len(errors) > 1 else ""
    return f"Invalid request at {loc}: {detail}.{more}"


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        body = error_body(
            exc.code, exc.message, retryable=exc.retryable, suggested_action=exc.suggested_action
        )
        return _response(request, exc.status_code, body, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        body = error_body(
            "VALIDATION_FAILED",
            _validation_message(exc),
            suggested_action="Correct the request and try again.",
        )
        return _response(request, 422, body)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {
            404: ("NOT_FOUND", "The resource was not found."),
            405: ("METHOD_NOT_ALLOWED", "This method is not allowed for this route."),
        }
        code, message = codes.get(exc.status_code, ("HTTP_ERROR", "The request failed."))
        return _response(request, exc.status_code, error_body(code, message))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error request_id=%s", _request_id_from(request))
        body = error_body(
            "INTERNAL",
            "An unexpected error occurred.",
            retryable=True,
            suggested_action="Try again later.",
        )
        return _response(request, 500, body)


class RequestContextMiddleware:
    """Assigns a request ID and marks every response as private/uncacheable."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        inbound = None
        for name, value in scope.get("headers", []):
            if name == b"x-request-id":
                inbound = value.decode("latin-1")
                break
        request_id = inbound if inbound and _SAFE_REQUEST_ID.match(inbound) else None
        request_id = request_id or secrets.token_hex(16)
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_var.set(request_id)

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if k.lower() not in (b"x-request-id", b"cache-control")
                ]
                headers.append((b"x-request-id", request_id.encode()))
                headers.append((b"cache-control", b"private, no-store"))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            request_id_var.reset(token)
