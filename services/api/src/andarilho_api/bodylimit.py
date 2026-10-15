"""The byte limit on layout saves, applied before the body is parsed.

``PUT /v1/scenes/{id}/layout`` is the one request whose contract-valid body can grow
large: the layout's counts allow far more than any home needs. A layout save is
therefore bounded in bytes too, at the size the web proxy forwards (the contract's
``MAX_LAYOUT_SAVE_BYTES``), so every layout the API accepts can be saved from the
browser.
"""

from __future__ import annotations

import re

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from andarilho_api.errors import error_body
from andarilho_api.schemas.scene import MAX_LAYOUT_SAVE_BYTES

LAYOUT_SAVE_PATH = re.compile(r"^/v1/scenes/[^/]+/layout$")


def _too_large(limit: int) -> JSONResponse:
    body = error_body(
        "PAYLOAD_TOO_LARGE",
        f"The layout is too large to save (at most {limit // (1024 * 1024)} MiB).",
        suggested_action="Remove some walls, obstacles or furniture, then save again.",
    )
    return JSONResponse(status_code=413, content=body)


class LayoutSizeLimitMiddleware:
    """Answers 413 to a layout save whose body is over ``limit`` bytes.

    The body is read here, within the limit, whether or not it declared a length, and the
    application is handed the same bytes.
    """

    def __init__(self, app: ASGIApp, limit: int = MAX_LAYOUT_SAVE_BYTES) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "PUT"
            or not LAYOUT_SAVE_PATH.match(scope["path"])
        ):
            await self.app(scope, receive, send)
            return
        body = bytearray()
        more = True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body += message.get("body", b"")
            if len(body) > self.limit:
                await _too_large(self.limit)(scope, receive, send)
                return
            more = message.get("more_body", False)
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)
