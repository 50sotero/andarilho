"""A small client for an OpenAI-compatible vision model that answers with one JSON object.

The worker asks the model about the capture's images (the ``scene_analysis`` stage). The
call is an ordinary bounded HTTP request: the key is read from the file that
``OPENCODE_API_KEY_FILE`` names each time a call is made, goes only into the
``Authorization`` header, and is never logged or put in an exception. Every failure is
reduced to a short reason that is safe to record, so the caller can carry on without the
model. The API's own AI client (``services/api``, its ``ai`` package) follows the same rules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from andarilho_worker.settings import Settings

USER_AGENT = "andarilho-worker"
# A reply is one JSON object of rooms, doors and furniture; anything larger is not one.
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_KEY_BYTES = 4096


class AIUnavailable(Exception):
    """The model could not answer. ``reason`` is one of a fixed set of safe words."""

    def __init__(self, reason: str, *, status: int | None = None) -> None:
        super().__init__(f"AI service unavailable: {reason}")
        self.reason = reason
        self.status = status


class AIInvalidReply(Exception):
    """The model answered, but not with a JSON object."""


@dataclass(frozen=True)
class Completion:
    data: dict[str, Any]
    model: str
    usage: dict[str, int]  # promptTokens, completionTokens, totalTokens when reported


def _json_text(content: str) -> str:
    """The JSON in a reply, without a Markdown code fence around it when the model adds one."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _usage(raw: object) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    names = {
        "prompt_tokens": "promptTokens",
        "completion_tokens": "completionTokens",
        "total_tokens": "totalTokens",
    }
    usage: dict[str, int] = {}
    for source, name in names.items():
        value = raw.get(source)
        if isinstance(value, int) and not isinstance(value, bool):
            usage[name] = value
    return usage


def _status_error(status: int) -> AIUnavailable:
    if status == 402:
        reason = "no_credit"
    elif status in (401, 403):
        reason = "auth"
    elif status == 429:
        reason = "rate_limited"
    elif status >= 500:
        reason = "server"
    else:
        reason = "bad_status"
    return AIUnavailable(reason, status=status)


def _read_capped(response: httpx.Response) -> bytes:
    raw = bytearray()
    for chunk in response.iter_bytes():
        raw += chunk
        if len(raw) > MAX_RESPONSE_BYTES:
            raise AIInvalidReply("the AI reply is too large")
    return bytes(raw)


class VisionClient:
    """One OpenAI-compatible chat completion per call, with JSON output requested."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        key_file: str | None,
        timeout_seconds: float,
        max_output_tokens: int = 4096,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._key_file = key_file
        self._timeout = httpx.Timeout(timeout_seconds)
        self._max_output_tokens = max_output_tokens
        self._transport = transport

    @classmethod
    def from_settings(cls, settings: Settings) -> VisionClient:
        return cls(
            base_url=settings.ai_base_url,
            model=settings.ai_model,
            key_file=settings.opencode_api_key_file,
            timeout_seconds=settings.ai_timeout_seconds,
            max_output_tokens=settings.ai_max_output_tokens,
        )

    def complete_json(self, system: str, content: list[dict[str, Any]]) -> Completion:
        """Send ``system`` and ``content`` (OpenAI content parts) and return the JSON reply.

        ``content`` holds text parts and ``image_url`` parts whose URLs are data URLs.
        Raises :class:`AIUnavailable` when the model cannot answer and
        :class:`AIInvalidReply` when its reply is not one JSON object.
        """
        key = self._read_key()
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "max_tokens": self._max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        raw = self._post(key, body)
        return self._completion(raw)

    def _read_key(self) -> str:
        if not self._key_file:
            raise AIUnavailable("no_key")
        path = Path(self._key_file)
        try:
            if path.stat().st_size > MAX_KEY_BYTES:
                raise AIUnavailable("no_key")
            value = path.read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeDecodeError):
            raise AIUnavailable("no_key") from None
        if not value or any(ch.isspace() for ch in value):
            raise AIUnavailable("no_key")
        return value

    def _post(self, key: str, body: dict[str, Any]) -> bytes:
        headers = {"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT}
        try:
            with (
                httpx.Client(
                    transport=self._transport, timeout=self._timeout, follow_redirects=False
                ) as client,
                client.stream("POST", self._url, json=body, headers=headers) as response,
            ):
                if response.status_code != 200:
                    raise _status_error(response.status_code)
                return _read_capped(response)
        except httpx.TimeoutException:
            raise AIUnavailable("timeout") from None
        except httpx.TransportError:
            raise AIUnavailable("network") from None
        except httpx.HTTPError:
            # A body that cannot be decoded (for example a corrupt compressed reply).
            raise AIInvalidReply("the AI reply could not be read") from None

    def _completion(self, raw: bytes) -> Completion:
        try:
            payload = json.loads(raw)
        except ValueError:
            raise AIInvalidReply("the AI service did not return JSON") from None
        if not isinstance(payload, dict):
            raise AIInvalidReply("the AI service did not return a JSON object")
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise AIInvalidReply("the AI reply has no message") from None
        if not isinstance(content, str):
            raise AIInvalidReply("the AI reply has no text")
        try:
            data = json.loads(_json_text(content))
        except ValueError:
            raise AIInvalidReply("the AI reply is not JSON") from None
        if not isinstance(data, dict):
            raise AIInvalidReply("the AI reply is not a JSON object")
        return Completion(data=data, model=self._model, usage=_usage(payload.get("usage")))
