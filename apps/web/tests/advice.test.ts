import { describe, expect, it, vi } from "vitest";
import { keepOffering } from "@/features/processing/RunAdvice";
import { createApiClient } from "@/lib/api/client";
import { ApiError, toFriendlyError } from "@/lib/api/errors";

const ADVICE = {
  runId: "r1",
  provider: "opencode",
  model: "deepseek-v4.1-flash",
  advice: { summary: "The walk was too fast.", steps: ["Walk slowly."] },
  sent: "The error code, its message and capture statistics. No images or files are sent.",
};

function aiError(code: string, retryable: boolean) {
  return new ApiError(code === "AI_RATE_LIMITED" ? 429 : 503, {
    code,
    message: "x",
    retryable,
  });
}

describe("run advice", () => {
  it("asks the API for advice on one run", async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify(ADVICE), { status: 200 }));
    const { data } = await createApiClient(fetchMock).getRunAdvice("run/1");
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/backend/v1/runs/run%2F1/advice");
    expect(init.method).toBe("POST");
    expect(data.advice.steps).toEqual(["Walk slowly."]);
  });

  it("stops offering only when AI is not set up", () => {
    expect(keepOffering(aiError("AI_UNAVAILABLE", false))).toBe(false);
    expect(keepOffering(aiError("AI_UNAVAILABLE", true))).toBe(true); // provider did not answer
    expect(keepOffering(aiError("AI_INVALID_OUTPUT", true))).toBe(true);
    expect(keepOffering(new TypeError("network"))).toBe(true);
  });

  it("tells the user how long to wait when the provider is busy", () => {
    const busy = new ApiError(
      429,
      { code: "AI_RATE_LIMITED", message: "The AI provider is busy.", retryable: true },
      { retryAfterSeconds: 7 },
    );
    expect(toFriendlyError(busy)).toMatchObject({
      title: "The AI provider is busy",
      suggestedAction: "Wait about 7 seconds and try again.",
    });
    expect(toFriendlyError(aiError("AI_UNAVAILABLE", false)).title).toBe(
      "AI suggestions aren't available",
    );
  });
});
