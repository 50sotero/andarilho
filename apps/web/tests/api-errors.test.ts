import { describe, expect, it, vi } from "vitest";
import { createApiClient } from "@/lib/api/client";
import {
  ApiError,
  apiErrorFromResponse,
  friendlyFromErrorBody,
  toFriendlyError,
} from "@/lib/api/errors";

function envelope(
  status: number,
  error: Record<string, unknown>,
  headers: Record<string, string> = {},
) {
  return new Response(JSON.stringify({ error }), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

describe("apiErrorFromResponse", () => {
  it("parses the structured error envelope", async () => {
    const err = await apiErrorFromResponse(
      envelope(409, {
        code: "VERSION_CONFLICT",
        message: "The scene changed since you loaded it.",
        retryable: false,
        suggestedAction: "Reload the scene and apply your edit again.",
        requestId: "req-9",
      }),
    );
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({
      status: 409,
      code: "VERSION_CONFLICT",
      retryable: false,
      suggestedAction: "Reload the scene and apply your edit again.",
      requestId: "req-9",
    });
  });

  it("reads Retry-After", async () => {
    const err = await apiErrorFromResponse(
      envelope(
        429,
        { code: "RATE_LIMITED", message: "Slow down.", retryable: true },
        { "Retry-After": "12" },
      ),
    );
    expect(err.retryAfterSeconds).toBe(12);
    expect(toFriendlyError(err).suggestedAction).toBe("Wait about 12 seconds and try again.");
  });

  it("falls back for non-envelope bodies", async () => {
    const err = await apiErrorFromResponse(
      new Response("<html>bad gateway</html>", { status: 502 }),
    );
    expect(err.code).toBe("SERVER_ERROR");
    expect(err.retryable).toBe(true);
    expect(err.message).not.toContain("html");
    const notFound = await apiErrorFromResponse(new Response("", { status: 404 }));
    expect(notFound.code).toBe("NOT_FOUND");
  });
});

describe("toFriendlyError", () => {
  it("maps stable codes to a title and prefers the API's suggested action", () => {
    const friendly = friendlyFromErrorBody({
      code: "DISCONNECTED_CAPTURE",
      message: "The bedroom and hallway could not be connected.",
      retryable: false,
      suggestedAction: "Record a slow continuous walk through the doorway.",
      requestId: "r1",
    });
    expect(friendly.title).toBe("Parts of the capture couldn't be connected");
    expect(friendly.message).toBe("The bedroom and hallway could not be connected.");
    expect(friendly.suggestedAction).toBe("Record a slow continuous walk through the doorway.");
    expect(friendly.requestId).toBe("r1");
  });

  it("uses a fallback action when the API gives none", () => {
    const friendly = toFriendlyError(
      new ApiError(409, {
        code: "ACTIVE_RUN_EXISTS",
        message: "A run is active.",
        retryable: false,
      }),
    );
    expect(friendly.title).toBe("A reconstruction is already running");
    expect(friendly.suggestedAction).toMatch(/cancel/i);
  });

  it("handles unknown codes, network errors and arbitrary values", () => {
    expect(
      toFriendlyError(new ApiError(500, { code: "WEIRD", message: "x", retryable: true })).title,
    ).toBe("Something went wrong");
    expect(toFriendlyError(new TypeError("Failed to fetch"))).toMatchObject({
      title: "Network problem",
      retryable: true,
    });
    expect(toFriendlyError("nope").title).toBe("Something went wrong");
  });
});

describe("api client", () => {
  it("always calls the same-origin proxy with the right headers", async () => {
    const fetchMock = vi.fn(
      async (_input: string, _init?: RequestInit) =>
        new Response(
          JSON.stringify({ runId: "r", jobId: "j", state: "queued", stage: "validating" }),
          {
            status: 202,
            headers: { ETag: '"2"' },
          },
        ),
    );
    const client = createApiClient(fetchMock);
    const result = await client.submitRun(
      "p/1",
      { captureSetId: "c", quality: "preview", includeSupplementaryPhotos: true },
      "idem-1",
    );
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/backend/v1/projects/p%2F1/runs");
    const headers = init!.headers as Record<string, string>;
    expect(headers["Idempotency-Key"]).toBe("idem-1");
    expect(headers["Content-Type"]).toBe("application/json");
    expect(init!.credentials).toBe("same-origin");
    expect(result).toEqual({
      data: { runId: "r", jobId: "j", state: "queued", stage: "validating" },
      etag: '"2"',
    });
  });

  it("sends If-Match and handles 204", async () => {
    const fetchMock = vi.fn(
      async (_input: string, _init?: RequestInit) => new Response(null, { status: 204 }),
    );
    const client = createApiClient(fetchMock);
    await client.revokeShare("s1");
    await client.renameProject("p1", "New", '"7"').catch(() => undefined);
    expect((fetchMock.mock.calls[1]![1]!.headers as Record<string, string>)["If-Match"]).toBe(
      '"7"',
    );
  });

  it("omits empty query values", async () => {
    const fetchMock = vi.fn(
      async (_input: string, _init?: RequestInit) =>
        new Response(JSON.stringify({ items: [], nextCursor: null }), { status: 200 }),
    );
    await createApiClient(fetchMock).listProjects(null, 20);
    expect(fetchMock.mock.calls[0]![0]).toBe("/api/backend/v1/projects?limit=20");
  });

  it("throws ApiError on error responses", async () => {
    const client = createApiClient(async () =>
      envelope(404, { code: "NOT_FOUND", message: "Not found.", retryable: false }),
    );
    await expect(client.getProject("x")).rejects.toMatchObject({ code: "NOT_FOUND", status: 404 });
  });
});
