"""The worker's OpenAI-compatible vision client, against httpx.MockTransport (no network)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from andarilho_worker.ai import (
    MAX_RESPONSE_BYTES,
    AIInvalidReply,
    AIUnavailable,
    Completion,
    VisionClient,
)
from andarilho_worker.settings import Settings

BASE = "https://gateway.test/v1"
KEY = "sk-test-0123456789abcdef"
CONTENT: list[dict[str, Any]] = [
    {"type": "text", "text": "Describe the room."},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
]


def _reply(content: Any, usage: dict[str, Any] | None = None) -> httpx.Response:
    body: dict[str, Any] = {
        "id": "chatcmpl-1",
        "model": "deepseek-v4.1-flash",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
    }
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def _key_file(tmp_path: Path, text: str = KEY) -> Path:
    path = tmp_path / "ai-key"
    path.write_text(text, encoding="utf-8")
    return path


def _client(
    key_file: Path | None,
    handler,
    *,
    base_url: str = BASE,
    timeout_seconds: float = 30.0,
) -> VisionClient:
    return VisionClient(
        base_url=base_url,
        model="deepseek-v4.1-flash",
        key_file=None if key_file is None else str(key_file),
        timeout_seconds=timeout_seconds,
        transport=httpx.MockTransport(handler),
    )


def _fail(client: VisionClient) -> AIUnavailable:
    with pytest.raises(AIUnavailable) as caught:
        client.complete_json("system", CONTENT)
    return caught.value


def _never_called(request: httpx.Request) -> httpx.Response:
    raise AssertionError("no request should be sent")


# --------------------------------------------------------------------------
# Successful replies
# --------------------------------------------------------------------------


def test_a_json_reply_comes_back_with_its_usage(tmp_path):
    usage = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}
    client = _client(_key_file(tmp_path), lambda request: _reply('{"rooms": []}', usage))
    completion = client.complete_json("system", CONTENT)
    assert completion == Completion(
        data={"rooms": []},
        model="deepseek-v4.1-flash",
        usage={"promptTokens": 120, "completionTokens": 30, "totalTokens": 150},
    )


def test_a_reply_without_usage_reports_an_empty_usage(tmp_path):
    client = _client(_key_file(tmp_path), lambda request: _reply('{"frames": []}'))
    assert client.complete_json("system", CONTENT).usage == {}


@pytest.mark.parametrize(
    "content",
    [
        '```json\n{"rooms": []}\n```',
        '```\n{"rooms": []}\n```',
        '  \n{"rooms": []}\n  ',
        '```json\n{"rooms": []}',
    ],
)
def test_a_markdown_fence_or_surrounding_space_is_tolerated(tmp_path, content):
    client = _client(_key_file(tmp_path), lambda request: _reply(content))
    assert client.complete_json("system", CONTENT).data == {"rooms": []}


def test_the_request_is_a_json_chat_call_with_the_key_in_the_header_only(tmp_path):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _reply("{}")

    _client(_key_file(tmp_path), handler, base_url=BASE + "/").complete_json("system", CONTENT)
    (request,) = seen
    assert request.method == "POST"
    assert str(request.url) == f"{BASE}/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.headers["user-agent"] == "andarilho-worker"
    body = json.loads(request.content)
    assert body == {
        "model": "deepseek-v4.1-flash",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": CONTENT},
        ],
        "temperature": 0,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }
    assert KEY not in request.content.decode()


def test_the_key_is_read_from_the_file_at_each_call(tmp_path):
    keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        keys.append(request.headers["authorization"])
        return _reply("{}")

    path = _key_file(tmp_path, KEY)
    client = _client(path, handler)
    client.complete_json("system", CONTENT)
    path.write_text("sk-rotated-9876543210", encoding="utf-8")
    client.complete_json("system", CONTENT)
    assert keys == [f"Bearer {KEY}", "Bearer sk-rotated-9876543210"]


def test_the_key_file_is_stripped_of_surrounding_whitespace(tmp_path):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return _reply("{}")

    client = _client(_key_file(tmp_path, f"﻿  {KEY}\r\n\n"), handler)
    client.complete_json("system", CONTENT)
    assert seen == [f"Bearer {KEY}"]


# --------------------------------------------------------------------------
# Unavailable: the model could not answer
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (402, "no_credit"),
        (401, "auth"),
        (403, "auth"),
        (429, "rate_limited"),
        (500, "server"),
        (503, "server"),
        (404, "bad_status"),
        (400, "bad_status"),
        (302, "bad_status"),
    ],
)
def test_each_status_maps_to_one_safe_reason(tmp_path, status, reason):
    body = f"quota details for {KEY}"
    client = _client(_key_file(tmp_path), lambda request: httpx.Response(status, text=body))
    error = _fail(client)
    assert error.reason == reason
    assert error.status == status
    assert KEY not in str(error) and KEY not in repr(error)


def test_a_timeout_is_reported_as_timeout(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    error = _fail(_client(_key_file(tmp_path), handler))
    assert error.reason == "timeout" and error.status is None


def test_other_transport_errors_are_reported_as_network(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    error = _fail(_client(_key_file(tmp_path), handler))
    assert error.reason == "network" and error.status is None


def test_a_missing_key_file_is_no_key_and_sends_nothing(tmp_path):
    error = _fail(_client(tmp_path / "absent", _never_called))
    assert error.reason == "no_key"


def test_no_key_file_setting_is_no_key_and_sends_nothing(tmp_path):
    error = _fail(_client(None, _never_called))
    assert error.reason == "no_key"


@pytest.mark.parametrize("text", ["", "   \n", "sk-with a space", "\n"])
def test_an_empty_or_malformed_key_file_is_no_key(tmp_path, text):
    error = _fail(_client(_key_file(tmp_path, text), _never_called))
    assert error.reason == "no_key"


def test_an_oversized_key_file_is_no_key(tmp_path):
    error = _fail(_client(_key_file(tmp_path, "k" * 5000), _never_called))
    assert error.reason == "no_key"


def test_a_key_file_that_is_not_text_is_no_key(tmp_path):
    path = tmp_path / "ai-key"
    path.write_bytes(b"\xff\xfe\x00 bad")
    assert _fail(_client(path, _never_called)).reason == "no_key"


# --------------------------------------------------------------------------
# Invalid: the model answered, but not with one JSON object
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "respond",
    [
        lambda request: _reply("this is not json"),
        lambda request: _reply("[1, 2, 3]"),
        lambda request: _reply('"just a string"'),
        lambda request: _reply(None),
        lambda request: _reply(42),
        lambda request: httpx.Response(200, content=b"<html>not json</html>"),
        lambda request: httpx.Response(200, json=[1, 2]),
        lambda request: httpx.Response(200, json={"choices": []}),
        lambda request: httpx.Response(200, json={"model": "x"}),
        lambda request: httpx.Response(200, content=b" " * (MAX_RESPONSE_BYTES + 1)),
    ],
)
def test_an_unusable_reply_is_invalid(tmp_path, respond):
    client = _client(_key_file(tmp_path), respond)
    with pytest.raises(AIInvalidReply) as caught:
        client.complete_json("system", CONTENT)
    assert KEY not in str(caught.value) and KEY not in repr(caught.value)


def test_a_reply_at_the_size_cap_is_still_read(tmp_path):
    # Whitespace inside the JSON pads the body up to the cap without changing the reply.
    base = json.dumps({"choices": [{"message": {"content": '{"rooms": []}'}}]}).encode()
    body = base[:1] + b" " * (MAX_RESPONSE_BYTES - len(base)) + base[1:]
    assert len(body) == MAX_RESPONSE_BYTES
    client = _client(_key_file(tmp_path), lambda request: httpx.Response(200, content=body))
    assert client.complete_json("system", CONTENT).data == {"rooms": []}


def test_an_undecodable_body_is_invalid(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.DecodingError("corrupt compressed body", request=request)

    client = _client(_key_file(tmp_path), handler)
    with pytest.raises(AIInvalidReply):
        client.complete_json("system", CONTENT)


def test_an_invalid_reply_never_echoes_the_key(tmp_path):
    client = _client(_key_file(tmp_path), lambda request: _reply(f"oops {KEY} {{"))
    with pytest.raises(AIInvalidReply) as caught:
        client.complete_json("system", CONTENT)
    assert KEY not in str(caught.value) and KEY not in repr(caught.value)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_from_settings_takes_the_worker_ai_settings(tmp_path):
    settings = dataclasses.replace(
        Settings(database_url="postgresql+psycopg://u:p@localhost:5432/db"),
        ai_base_url="https://gateway.test/v1/",
        ai_model="model-x",
        opencode_api_key_file=str(tmp_path / "key"),
        ai_timeout_seconds=45.0,
    )
    client = VisionClient.from_settings(settings)
    assert client._url == "https://gateway.test/v1/chat/completions"
    assert client._model == "model-x"
    assert client._key_file == str(tmp_path / "key")
    assert client._timeout.read == 45.0
