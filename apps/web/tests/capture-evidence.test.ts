import { describe, expect, it, vi } from "vitest";
import {
  EVIDENCE_MAX_ITEMS,
  evidenceOnPlan,
  evidenceScale,
  loadCaptureEvidence,
  parseCaptureEvidence,
  type CaptureEvidence,
} from "@/features/review/capture-evidence";

const valid = {
  schemaVersion: 1,
  units: "m",
  points: [[1, 0.5, 2]],
  cameras: [[0, 1.5, 0]],
};

const later = () => new Date(Date.now() + 60_000).toISOString();

function jsonResponse(body: unknown, ok = true): Response {
  return { ok, json: async () => body } as unknown as Response;
}

describe("parseCaptureEvidence", () => {
  it("accepts a version 1 file in metres, including empty lists", () => {
    expect(parseCaptureEvidence(valid)).toEqual({
      points: [[1, 0.5, 2]],
      cameras: [[0, 1.5, 0]],
    });
    expect(parseCaptureEvidence({ ...valid, points: [], cameras: [] })).toEqual({
      points: [],
      cameras: [],
    });
  });

  it("rejects an unknown schema version or unit", () => {
    expect(parseCaptureEvidence({ ...valid, schemaVersion: 2 })).toBeNull();
    expect(parseCaptureEvidence({ ...valid, schemaVersion: "1" })).toBeNull();
    expect(parseCaptureEvidence({ ...valid, units: "ft" })).toBeNull();
  });

  it("rejects values that are not an object", () => {
    for (const bad of [null, undefined, "evidence", 42, true]) {
      expect(parseCaptureEvidence(bad)).toBeNull();
    }
  });

  it("rejects missing or non-array point and camera lists", () => {
    expect(parseCaptureEvidence({ ...valid, points: undefined })).toBeNull();
    expect(parseCaptureEvidence({ ...valid, cameras: { 0: [0, 0, 0] } })).toBeNull();
    expect(parseCaptureEvidence({ ...valid, points: "[1,2,3]" })).toBeNull();
  });

  it("rejects vectors that are not three finite numbers", () => {
    for (const bad of [
      [1, 2],
      [1, 2, 3, 4],
      [1, "2", 3],
      [1, null, 3],
      [1, Number.NaN, 3],
      [1, Number.POSITIVE_INFINITY, 3],
      { x: 1, y: 2, z: 3 },
      "1,2,3",
    ]) {
      expect(parseCaptureEvidence({ ...valid, points: [bad] })).toBeNull();
      expect(parseCaptureEvidence({ ...valid, cameras: [bad] })).toBeNull();
    }
  });

  it("rejects the whole file when one point is malformed", () => {
    const mixed = {
      ...valid,
      points: [
        [1, 0.5, 2],
        [1, 0.5],
      ],
    };
    expect(parseCaptureEvidence(mixed)).toBeNull();
  });

  it("caps each list at the item limit", () => {
    const vectors = (n: number) => Array.from({ length: n }, (_, i) => [i, 1, i]);
    expect(
      parseCaptureEvidence({ ...valid, points: vectors(EVIDENCE_MAX_ITEMS) })?.points,
    ).toHaveLength(EVIDENCE_MAX_ITEMS);
    expect(parseCaptureEvidence({ ...valid, points: vectors(EVIDENCE_MAX_ITEMS + 1) })).toBeNull();
    expect(parseCaptureEvidence({ ...valid, cameras: vectors(EVIDENCE_MAX_ITEMS + 1) })).toBeNull();
  });
});

describe("evidenceScale", () => {
  it("is 1 before the user calibrates", () => {
    expect(evidenceScale(undefined)).toBe(1);
    expect(evidenceScale(null)).toBe(1);
  });

  it("uses the cumulative render scale of the calibration", () => {
    expect(evidenceScale({ method: "known_distance", renderScale: 2.5 })).toBe(2.5);
  });

  it("treats a scale that is not a positive finite number as uncalibrated", () => {
    for (const renderScale of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(evidenceScale({ method: "known_distance", renderScale })).toBe(1);
    }
  });
});

describe("evidenceOnPlan", () => {
  it("scales the horizontal position by the render scale and drops the height", () => {
    const plan = evidenceOnPlan({ points: [[1, 0.5, 2]], cameras: [[3, 1.5, 4]] }, 2);
    expect(plan.points).toEqual([[2, 4]]);
    expect(plan.cameraPath).toEqual([[6, 8]]);
  });

  it("keeps only points between 0.1 m and 2.2 m above the floor", () => {
    const evidence: CaptureEvidence = {
      points: [
        [0, 0.05, 0], // floor clutter
        [1, 0.1, 0], // at the lower edge
        [2, 1.0, 0],
        [3, 2.2, 0], // at the upper edge
        [4, 2.5, 0], // ceiling clutter
      ],
      cameras: [],
    };
    expect(evidenceOnPlan(evidence, 1).points.map(([x]) => x)).toEqual([1, 2, 3]);
  });

  it("applies the height band to calibrated heights, not the stored ones", () => {
    // The stored y is in the estimated frame; at scale 2 it is 0.12 m and 2.0 m on the plan.
    const evidence: CaptureEvidence = {
      points: [
        [0, 0.06, 0],
        [1, 0.04, 0],
        [2, 1.0, 0],
        [3, 1.2, 0],
      ],
      cameras: [],
    };
    expect(evidenceOnPlan(evidence, 2).points.map(([x]) => x)).toEqual([0, 4]);
  });

  it("keeps every camera position in path order", () => {
    const cameras: CaptureEvidence["cameras"] = [
      [0, 1.5, 0],
      [0, 1.5, 1],
      [0, 1.5, 2],
    ];
    const plan = evidenceOnPlan({ points: [], cameras }, 1);
    expect(plan.cameraPath).toEqual([
      [0, 0],
      [0, 1],
      [0, 2],
    ]);
  });
});

describe("loadCaptureEvidence", () => {
  const link = { url: "https://storage.test/evidence.json", expiresAt: later() };

  it("downloads the link without credentials and returns the parsed evidence", async () => {
    const fetchMock = vi.fn<typeof fetch>(async () => jsonResponse(valid));
    await expect(loadCaptureEvidence(link, fetchMock)).resolves.toEqual({
      points: [[1, 0.5, 2]],
      cameras: [[0, 1.5, 0]],
    });
    expect(fetchMock).toHaveBeenCalledWith(link.url, { credentials: "omit" });
  });

  it("resolves to null, quietly, when the request fails or the file is malformed", async () => {
    const refused = vi.fn<typeof fetch>(async () => jsonResponse({}, false));
    const offline = vi.fn<typeof fetch>(async () => {
      throw new TypeError("Failed to fetch");
    });
    const malformed = vi.fn<typeof fetch>(async () => jsonResponse({ ...valid, units: "cm" }));
    const notJson = vi.fn<typeof fetch>(
      async () =>
        ({
          ok: true,
          json: async () => {
            throw new SyntaxError("Unexpected token");
          },
        }) as unknown as Response,
    );
    for (const fetchMock of [refused, offline, malformed, notJson]) {
      await expect(loadCaptureEvidence(link, fetchMock)).resolves.toBeNull();
    }
  });

  it("does not download a link that has already expired", async () => {
    const fetchMock = vi.fn<typeof fetch>(async () => jsonResponse(valid));
    const expired = { url: link.url, expiresAt: new Date(Date.now() - 1000).toISOString() };
    await expect(loadCaptureEvidence(expired, fetchMock)).resolves.toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
