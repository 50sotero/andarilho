"""Optional AI advice for failed runs: bounded, sanitized, schema-checked, never required."""

from __future__ import annotations

import json
import uuid

import httpx
import pytest
from factories import active_run, create_project
from pydantic import SecretStr
from sqlalchemy import select

from andarilho_api.ai.client import AIClient, AIError, read_api_key
from andarilho_api.main import create_app
from andarilho_api.models import AuditEvent, Job, ReconstructionRun

KEY = "test-key-0123456789"
ADVICE = {"summary": "The walk moved too fast for the views to overlap.", "steps": ["Walk slowly."]}
FAILURE = {
    "code": "INSUFFICIENT_OVERLAP",
    "message": "Too few views overlapped to reconstruct the place reliably.",
    "suggestedAction": "Record again, moving slowly.",
    "details": {
        "registeredViews": 28,
        "selectedViews": 141,
        "disconnectedComponents": 8,
        "registeredRatio": 0.1986,
        "scratchPath": "/scratch/job/images",  # never sent: not a capture statistic
    },
}


def reply(content: str, status: int = 200, **headers: str) -> httpx.Response:
    body = {"choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 210, "completion_tokens": 64}}  # fmt: skip
    return httpx.Response(status, json=body, headers=headers)


class Provider:
    """A mock OpenAI-compatible endpoint: queued responses, recorded requests."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0) if self.responses else reply(json.dumps(ADVICE))


@pytest.fixture
def key_file(tmp_path):
    path = tmp_path / "opencode"
    path.write_text(f"  {KEY}\n", encoding="utf-8")
    return path


@pytest.fixture
def ai_settings(settings, key_file):
    return settings.model_copy(
        update={
            "ai_provider": "opencode",
            "ai_base_url": "https://ai.test.invalid/v1/",
            "opencode_api_key_file": str(key_file),
            "ai_requests_per_minute": 3,
        }
    )


@pytest.fixture
def provider():
    return Provider()


@pytest.fixture
def ai_client(aws, ai_settings, keys, provider):
    from fastapi.testclient import TestClient

    app = create_app(
        ai_settings,
        jwks_fetcher=keys.fetch,
        ai_client=AIClient(ai_settings, transport=httpx.MockTransport(provider)),
    )
    with TestClient(app) as test_client:
        yield test_client
    app.state.engine.dispose()


def failed_run(db, client, auth) -> str:
    pid = create_project(client, auth)["id"]
    info = active_run(db, pid, job_states=("succeeded", "succeeded", "failed"))
    run = db.get(ReconstructionRun, uuid.UUID(info["runId"]))
    run.status, run.error = "failed", FAILURE
    db.commit()
    return info["runId"]


def test_failed_run_advice_sends_only_sanitized_facts(ai_client, auth, db, provider):
    run_id = failed_run(db, ai_client, auth)
    provider.responses.append(reply("```json\n" + json.dumps(ADVICE) + "\n```"))
    response = ai_client.post(f"/v1/runs/{run_id}/advice", headers=auth())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["advice"] == ADVICE
    assert body["provider"] == "opencode" and body["model"] == "deepseek-v4.1-flash"
    assert "No images or files are sent" in body["sent"]

    (request,) = provider.requests
    assert str(request.url) == "https://ai.test.invalid/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"  # trimmed from the file
    sent = json.loads(request.content)
    assert sent["model"] == "deepseek-v4.1-flash" and sent["max_tokens"] == 2048
    system, user = sent["messages"]
    assert system["role"] == "system" and "untrusted data" in system["content"]
    assert "INSUFFICIENT_OVERLAP" in user["content"]
    assert '"failedStage": "camera_reconstruction"' in user["content"]
    assert '"registeredViews": 28' in user["content"]
    assert "scratch" not in request.content.decode() and KEY not in request.content.decode()

    event = db.execute(select(AuditEvent).where(AuditEvent.action == "run.ai_advice")).scalar_one()
    assert event.data == {"provider": "opencode", "model": "deepseek-v4.1-flash",
                          "promptTokens": 210, "completionTokens": 64}  # fmt: skip


def test_failure_prompt_describes_camera_reconstruction_and_texturing(
    ai_client, auth, db, provider
):
    run_id = failed_run(db, ai_client, auth)
    assert ai_client.post(f"/v1/runs/{run_id}/advice", headers=auth()).status_code == 200
    (request,) = provider.requests
    prompt = json.loads(request.content)["messages"][1]["content"]
    assert "camera reconstruction (COLMAP)" in prompt and "textured" in prompt
    assert "Gaussian" not in prompt


def test_advice_is_unavailable_unless_configured(client, auth, db):
    run_id = failed_run(db, client, auth)
    response = client.post(f"/v1/runs/{run_id}/advice", headers=auth())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "AI_UNAVAILABLE"
    assert response.json()["error"]["retryable"] is False


def test_advice_only_for_failed_runs_and_members(ai_client, auth, db, provider):
    pid = create_project(ai_client, auth)["id"]
    running = active_run(db, pid, job_states=("running",))
    assert ai_client.post(f"/v1/runs/{running['runId']}/advice", headers=auth()).status_code == 409
    run_id = failed_run(db, ai_client, auth)
    assert ai_client.post(f"/v1/runs/{run_id}/advice", headers=auth("user-b")).status_code == 404
    assert provider.requests == []


@pytest.mark.parametrize(
    ("responses", "status", "code", "calls"),
    [
        ([reply("", 401)], 503, "AI_UNAVAILABLE", 1),
        ([reply("", 429, **{"Retry-After": "7"})], 429, "AI_RATE_LIMITED", 1),
        ([reply("", 502), reply("", 503)], 503, "AI_UNAVAILABLE", 2),  # one retry
        ([reply("not json")], 502, "AI_INVALID_OUTPUT", 1),
        ([reply(json.dumps({"summary": "x", "steps": []}))], 502, "AI_INVALID_OUTPUT", 1),
    ],
)
def test_provider_failures_map_to_distinct_errors(
    ai_client, auth, db, provider, responses, status, code, calls
):
    run_id = failed_run(db, ai_client, auth)
    provider.responses.extend(responses)
    response = ai_client.post(f"/v1/runs/{run_id}/advice", headers=auth())
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert len(provider.requests) == calls
    if status == 429:
        assert response.headers["retry-after"] == "7"
    assert KEY not in response.text


def test_advice_requests_are_rate_limited_per_project(ai_client, auth, db, provider):
    run_id = failed_run(db, ai_client, auth)
    statuses = [ai_client.post(f"/v1/runs/{run_id}/advice", headers=auth()).status_code
                for _ in range(4)]  # fmt: skip
    assert statuses == [200, 200, 200, 429]
    assert len(provider.requests) == 3


def test_api_key_is_read_trimmed_and_never_echoed(settings, key_file, tmp_path):
    configured = settings.model_copy(update={"ai_provider": "opencode"})
    with pytest.raises(AIError) as missing:
        read_api_key(configured)
    assert missing.value.kind == "not_configured"
    from_file = configured.model_copy(update={"opencode_api_key_file": str(key_file)})
    assert read_api_key(from_file) == KEY
    from_env = from_file.model_copy(update={"opencode_api_key": SecretStr(" env-key ")})
    assert read_api_key(from_env) == "env-key"  # the environment wins over the file
    for content in ("", "two words"):
        bad = tmp_path / "bad"
        bad.write_text(content, encoding="utf-8")
        with pytest.raises(AIError) as exc:
            read_api_key(configured.model_copy(update={"opencode_api_key_file": str(bad)}))
        assert str(exc.value) == "not_configured" and "two" not in repr(exc.value)
    gone = configured.model_copy(update={"opencode_api_key_file": str(tmp_path / "missing")})
    with pytest.raises(AIError):
        read_api_key(gone)


def test_failed_job_stage_is_the_latest_failure(ai_client, auth, db, provider):
    run_id = failed_run(db, ai_client, auth)
    jobs = db.execute(select(Job).where(Job.run_id == uuid.UUID(run_id))).scalars().all()
    assert sorted(j.state for j in jobs) == ["failed", "succeeded", "succeeded"]
    assert ai_client.post(f"/v1/runs/{run_id}/advice", headers=auth()).status_code == 200
    user = json.loads(provider.requests[0].content)["messages"][1]["content"]
    assert '"failedStage": "camera_reconstruction"' in user
