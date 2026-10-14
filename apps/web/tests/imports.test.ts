import { describe, expect, it, vi } from "vitest";
import { listingHost, matterportEmbedUrl } from "@/features/imports/ImportsPanel";
import { createApiClient } from "@/lib/api/client";

describe("existing tours", () => {
  it("builds the embed only from a valid model id", () => {
    expect(matterportEmbedUrl("GzAxTe2zAQs")).toBe(
      "https://my.matterport.com/show/?m=GzAxTe2zAQs&play=0",
    );
    for (const bad of ["", "short", "GzAxTe2zAQs&x=1", "../../evil.x", "GzAxTe2zAQs1"]) {
      expect(matterportEmbedUrl(bad)).toBeNull();
    }
  });

  it("attributes a tour to its https listing host only", () => {
    expect(listingHost("https://www.homeq.se/lagenhet/1")).toBe("www.homeq.se");
    expect(listingHost("javascript:alert(1)")).toBeNull();
    expect(listingHost("not a url")).toBeNull();
    expect(listingHost(null)).toBeNull();
  });

  it("resolves a link and attaches a tour with the user's confirmation", async () => {
    const fetchMock = vi.fn(async () => new Response("{}", { status: 200 }));
    const client = createApiClient(fetchMock);
    await client.resolveImport("p1", "https://www.homeq.se/lagenhet/1");
    await client.createImport("p1", "GzAxTe2zAQs", "https://www.homeq.se/lagenhet/1");
    const calls = fetchMock.mock.calls as unknown as [string, RequestInit][];
    expect(calls[0]![0]).toBe("/api/backend/v1/projects/p1/imports/resolve");
    expect(JSON.parse(calls[0]![1].body as string)).toEqual({
      url: "https://www.homeq.se/lagenhet/1",
    });
    expect(JSON.parse(calls[1]![1].body as string)).toEqual({
      kind: "matterport_embed",
      modelId: "GzAxTe2zAQs",
      listingUrl: "https://www.homeq.se/lagenhet/1",
      permissionConfirmed: true,
    });
  });
});
