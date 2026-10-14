import { describe, expect, it, vi } from "vitest";
import {
  createBackendProxyHandler,
  isAllowedOrigin,
  validateProxyPath,
  type BackendProxyDeps,
} from "@/lib/api/backend-proxy";

const APP = "https://app.andarilho.test";
const API = "http://api:8000";

function setup(overrides: Partial<BackendProxyDeps> = {}) {
  const fetchMock = vi.fn(
    async (_input: string, _init: RequestInit) =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          ETag: '"3"',
          "X-Request-ID": "req-1",
          "Set-Cookie": "upstream=1",
          "Cache-Control": "public, max-age=600",
          Server: "uvicorn",
        },
      }),
  );
  const deps: BackendProxyDeps = {
    getAccessToken: vi.fn(async () => "token-abc"),
    getAppOrigin: () => APP,
    getApiBaseUrl: () => API,
    fetch: fetchMock,
    ...overrides,
  };
  return { handle: createBackendProxyHandler(deps), fetchMock, deps };
}

function req(path: string, init: RequestInit & { headers?: Record<string, string> } = {}) {
  return new Request(`${APP}/api/backend/${path}`, init);
}

describe("validateProxyPath", () => {
  it("accepts allowlisted /v1 resources", () => {
    expect(validateProxyPath(["v1", "projects"])).toEqual({ path: "v1/projects", isPublic: false });
    expect(validateProxyPath(["v1", "uploads", "abc", "parts"])?.path).toBe("v1/uploads/abc/parts");
    expect(validateProxyPath(["v1", "public", "scenes", "delivery"])).toEqual({
      path: "v1/public/scenes/delivery",
      isPublic: true,
    });
  });

  it("rejects non-/v1 and unknown paths", () => {
    expect(validateProxyPath(["healthz"])).toBeNull();
    expect(validateProxyPath(["readyz", "x"])).toBeNull();
    expect(validateProxyPath(["v2", "projects"])).toBeNull();
    expect(validateProxyPath(["v1", "admin"])).toBeNull();
    expect(validateProxyPath(["v1", "public", "scenes"])).toBeNull();
    expect(validateProxyPath(["v1", "public", "other"])).toBeNull();
    expect(validateProxyPath(["openapi.json"])).toBeNull();
    expect(validateProxyPath([])).toBeNull();
  });

  it("rejects traversal and encoded separators", () => {
    expect(validateProxyPath(["v1", "projects", "..", "..", "healthz"])).toBeNull();
    expect(validateProxyPath(["v1", "projects", "."])).toBeNull();
    expect(validateProxyPath(["v1", "projects", "a/b"])).toBeNull();
    expect(validateProxyPath(["v1", "projects", "a\\b"])).toBeNull();
    expect(validateProxyPath(["v1", "projects", "%2e%2e"])).toBeNull();
    expect(validateProxyPath(["v1", "projects", ""])).toBeNull();
    expect(validateProxyPath(["v1", "projects", "a?b"])).toBeNull();
  });
});

describe("isAllowedOrigin", () => {
  it("allows safe methods without an Origin", () => {
    expect(isAllowedOrigin("GET", null, APP)).toBe(true);
  });
  it("requires a matching Origin for mutations", () => {
    expect(isAllowedOrigin("POST", APP, APP)).toBe(true);
    expect(isAllowedOrigin("POST", null, APP)).toBe(false);
    expect(isAllowedOrigin("DELETE", "https://evil.test", APP)).toBe(false);
    expect(isAllowedOrigin("PUT", "null", APP)).toBe(false);
    expect(isAllowedOrigin("PATCH", "https://app.andarilho.test.evil.test", APP)).toBe(false);
  });
});

describe("backend proxy handler", () => {
  it("forwards GET with the session bearer token and query string", async () => {
    const { handle, fetchMock } = setup();
    const res = await handle(req("v1/projects?cursor=abc&limit=20"), ["v1", "projects"]);
    expect(res.status).toBe(200);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe(`${API}/v1/projects?cursor=abc&limit=20`);
    expect(new Headers(init.headers).get("authorization")).toBe("Bearer token-abc");
    expect(init.method).toBe("GET");
  });

  it("returns 404 for non-allowlisted paths without calling upstream", async () => {
    const { handle, fetchMock } = setup();
    for (const segments of [["healthz"], ["v1", "..", "healthz"], ["v1", "internal"]]) {
      const res = await handle(req(segments.join("/")), segments);
      expect(res.status).toBe(404);
      expect(res.headers.get("cache-control")).toBe("private, no-store");
      const body = await res.json();
      expect(body.error.code).toBe("NOT_FOUND");
    }
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects cross-origin and Origin-less mutations", async () => {
    const { handle, fetchMock } = setup();
    const evil = await handle(
      req("v1/projects", { method: "POST", headers: { Origin: "https://evil.test" }, body: "{}" }),
      ["v1", "projects"],
    );
    expect(evil.status).toBe(403);
    expect((await evil.json()).error.code).toBe("CROSS_ORIGIN_REJECTED");
    const missing = await handle(req("v1/projects/x", { method: "DELETE" }), ["v1", "projects", "x"]);
    expect(missing.status).toBe(403);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("forwards only allowlisted request headers and never client credentials", async () => {
    const { handle, fetchMock } = setup();
    await handle(
      req("v1/scenes/s1/layout", {
        method: "PUT",
        headers: {
          Origin: APP,
          "Content-Type": "application/json",
          "If-Match": '"4"',
          "Idempotency-Key": "k1",
          "X-Request-ID": "r1",
          Authorization: "Bearer attacker",
          Cookie: "sb-access-token=secret",
          "X-Owner-Id": "someone-else",
          "X-Forwarded-For": "1.2.3.4",
        },
        body: JSON.stringify({ baseRevisionId: "r", layout: {} }),
      }),
      ["v1", "scenes", "s1", "layout"],
    );
    const init = fetchMock.mock.calls[0]![1];
    const headers = new Headers(init.headers);
    expect(headers.get("authorization")).toBe("Bearer token-abc");
    expect(headers.get("content-type")).toBe("application/json");
    expect(headers.get("if-match")).toBe('"4"');
    expect(headers.get("idempotency-key")).toBe("k1");
    expect(headers.get("x-request-id")).toBe("r1");
    expect(headers.get("cookie")).toBeNull();
    expect(headers.get("x-owner-id")).toBeNull();
    expect(headers.get("x-forwarded-for")).toBeNull();
    expect(init.method).toBe("PUT");
    expect(new TextDecoder().decode(init.body as ArrayBuffer)).toContain("baseRevisionId");
  });

  it("forwards only allowlisted response headers and sets no-store", async () => {
    const { handle } = setup();
    const res = await handle(req("v1/projects/p1"), ["v1", "projects", "p1"]);
    expect(res.headers.get("cache-control")).toBe("private, no-store");
    expect(res.headers.get("etag")).toBe('"3"');
    expect(res.headers.get("x-request-id")).toBe("req-1");
    expect(res.headers.get("content-type")).toBe("application/json");
    expect(res.headers.get("set-cookie")).toBeNull();
    expect(res.headers.get("server")).toBeNull();
  });

  it("does not attach a token to the public share delivery route", async () => {
    const getAccessToken = vi.fn(async () => "token-abc");
    const { handle, fetchMock } = setup({ getAccessToken });
    const res = await handle(
      req("v1/public/scenes/delivery", {
        method: "POST",
        headers: { Origin: APP, "Content-Type": "application/json" },
        body: JSON.stringify({ token: "t" }),
      }),
      ["v1", "public", "scenes", "delivery"],
    );
    expect(res.status).toBe(200);
    expect(getAccessToken).not.toHaveBeenCalled();
    expect(new Headers(fetchMock.mock.calls[0]![1].headers).get("authorization")).toBeNull();
  });

  it("returns 401 without a session for private routes", async () => {
    const { handle, fetchMock } = setup({ getAccessToken: async () => null });
    const res = await handle(req("v1/projects"), ["v1", "projects"]);
    expect(res.status).toBe(401);
    expect((await res.json()).error.code).toBe("UNAUTHENTICATED");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("bounds the request body size", async () => {
    const { handle, fetchMock } = setup({ maxBodyBytes: 10 });
    const res = await handle(
      req("v1/projects", { method: "POST", headers: { Origin: APP }, body: "x".repeat(50) }),
      ["v1", "projects"],
    );
    expect(res.status).toBe(413);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("maps upstream connection failures to the error envelope", async () => {
    const { handle } = setup({
      fetch: async () => {
        throw new TypeError("connect ECONNREFUSED");
      },
    });
    const res = await handle(req("v1/projects"), ["v1", "projects"]);
    expect(res.status).toBe(502);
    expect(res.headers.get("cache-control")).toBe("private, no-store");
    const body = await res.json();
    expect(body.error).toMatchObject({ code: "UPSTREAM_UNAVAILABLE", retryable: true });
    expect(JSON.stringify(body)).not.toContain("ECONNREFUSED");
  });

  it("passes through upstream error statuses and 204 responses", async () => {
    const { handle } = setup({
      fetch: async () => new Response(null, { status: 204 }),
    });
    const res = await handle(
      req("v1/shares/s1", { method: "DELETE", headers: { Origin: APP } }),
      ["v1", "shares", "s1"],
    );
    expect(res.status).toBe(204);
    expect(await res.text()).toBe("");
  });

  it("does not follow or expose upstream redirects", async () => {
    const { handle } = setup({
      fetch: async () => new Response(null, { status: 302, headers: { Location: "http://internal/" } }),
    });
    const res = await handle(req("v1/projects"), ["v1", "projects"]);
    expect(res.status).toBe(502);
    expect(res.headers.get("location")).toBeNull();
  });
});
