import { describe, expect, it } from "vitest";
import { safeNextPath } from "@/lib/auth/safe-redirect";
import {
  calibrationFactor,
  isSelfIntersecting,
  layoutProblems,
  scaleLayout,
} from "@/features/review/geometry";
import { addOpening, movePoint, removeSelected } from "@/features/review/layout-edit";
import { canRetryRun, formatStageProgress, stageStates } from "@/features/processing/stages";
import { expiryFromDays } from "@/features/sharing/SharePanel";
import { fallbackLayout } from "@/features/projects/PrivateWalk";
import { removalMode } from "@/features/capture/CaptureFlow";
import { rejectionMessage } from "@/features/uploads/UploadList";
import type { DeliveryEnvelope, Layout, Run, SceneDetail } from "@/lib/api/types";
import fixture from "@andarilho/scene-contract/fixtures/synthetic-two-room.layout.json";

const layout = fixture as unknown as Layout;

describe("safeNextPath", () => {
  it("keeps same-origin relative paths", () => {
    expect(safeNextPath("/projects/abc?x=1#h")).toBe("/projects/abc?x=1#h");
  });
  it("rejects open-redirect attempts", () => {
    for (const bad of [
      "https://evil.test",
      "//evil.test",
      "/\\evil.test",
      "javascript:alert(1)",
      "evil",
      "/\u0000x",
      null,
      "",
    ]) {
      expect(safeNextPath(bad as string | null)).toBe("/projects");
    }
  });
});

describe("layout geometry", () => {
  it("accepts the synthetic fixture", () => {
    expect(layoutProblems(layout)).toEqual([]);
  });
  it("detects self-intersection", () => {
    expect(
      isSelfIntersecting([
        [0, 0],
        [1, 1],
        [1, 0],
        [0, 1],
      ]),
    ).toBe(true);
    expect(
      isSelfIntersecting([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
      ]),
    ).toBe(false);
  });
  it("flags a spawn outside the floor and a bad wall", () => {
    let edited = movePoint(layout, { kind: "spawn" }, [100, 100]);
    edited = {
      ...edited,
      walls: edited.walls!.map((w, i) => (i === 0 ? { ...w, thickness: -1 } : w)),
    };
    const messages = layoutProblems(edited)
      .map((p) => p.message)
      .join(" ");
    expect(messages).toMatch(/spawn/i);
    expect(messages).toMatch(/thickness/i);
  });
  it("calibrates uniformly", () => {
    expect(calibrationFactor(2, 3)).toBe(1.5);
    expect(calibrationFactor(0, 3)).toBeNull();
    const scaled = scaleLayout(layout, 2);
    expect(scaled.floors[0]!.polygons[0]![1]).toEqual([8, -5]);
    expect(scaled.walls![0]!.thickness).toBeCloseTo(0.2);
    expect(scaled.walls![0]!.height).toBeCloseTo(layout.walls![0]!.height * 2);
    expect(scaled.calibration).toEqual({ method: "known_distance", renderScale: 2 });
    // Calibration is cumulative and absolute, so the splats are never rescaled twice.
    expect(scaleLayout(scaled, 1.5).calibration?.renderScale).toBeCloseTo(3);
  });
  it("adds openings inside walls and removes a wall's openings with it", () => {
    const withOpening = addOpening(layout, 0);
    expect(layoutProblems(withOpening)).toEqual([]);
    const removed = removeSelected(withOpening, { kind: "wall", index: 0 });
    expect(removed.openings!.some((o) => o.wallId === layout.walls![0]!.id)).toBe(false);
  });
});

describe("run retry offer", () => {
  it("offers retry for infrastructure failures but not for bad captures", () => {
    const failed = (code: string) => ({
      status: "failed" as const,
      error: { code, message: "", retryable: false, suggestedAction: null, requestId: "r" },
    });
    expect(canRetryRun(failed("TRAINING_FAILED"))).toBe(true);
    expect(canRetryRun(failed("WORKER_LOST"))).toBe(true);
    expect(canRetryRun(failed("DISCONNECTED_CAPTURE"))).toBe(false);
    expect(canRetryRun({ status: "cancelled", error: null })).toBe(true);
    expect(canRetryRun({ status: "succeeded", error: null })).toBe(false);
  });
});

describe("processing helpers", () => {
  it("never invents progress", () => {
    expect(formatStageProgress(null)).toBeNull();
    expect(formatStageProgress({ message: "x" })).toBeNull();
    expect(formatStageProgress({ current: 3, total: 10, unit: "frames" })).toBe("3 of 10 frames");
    expect(formatStageProgress({ fraction: 0.25 })).toBe("25% of this stage");
  });
  it("derives stage states from jobs", () => {
    const run = {
      jobs: [
        { id: "1", stage: "validating", state: "succeeded" },
        { id: "2", stage: "preparing", state: "running" },
      ],
    } as unknown as Run;
    const states = stageStates(run);
    expect(states.validating).toBe("done");
    expect(states.preparing).toBe("active");
    expect(states.training).toBe("pending");
  });
});

describe("share expiry", () => {
  it("computes ISO expiry or null", () => {
    expect(expiryFromDays("", 0)).toBeNull();
    expect(expiryFromDays("1", 0)).toBe(new Date(86_400_000).toISOString());
  });
});

describe("private walk fallback plan", () => {
  const layout = fixture as unknown as Layout;
  const scene = { revision: { id: "rev-2", layout } } as unknown as SceneDetail;
  const envelopeFor = (id: string) => ({ sceneRevisionId: id }) as unknown as DeliveryEnvelope;

  it("shows the draft plan only when the draft is the walked revision", () => {
    expect(fallbackLayout(scene, envelopeFor("rev-2"))).toBe(layout);
    expect(fallbackLayout(scene, envelopeFor("rev-1"))).toBeNull();
    expect(fallbackLayout(scene, null)).toBe(layout);
    expect(fallbackLayout(null, envelopeFor("rev-1"))).toBeNull();
  });
});

describe("upload rejection message", () => {
  it("tells the user to re-upload when our check failed, not that the file is bad", () => {
    expect(rejectionMessage("INSPECTION_FAILED")).toMatch(/upload it again/);
    expect(rejectionMessage("INSPECTION_FAILED")).not.toMatch(/undamaged/);
    expect(rejectionMessage("UNSUPPORTED_MEDIA")).toMatch(/UNSUPPORTED_MEDIA.*undamaged/);
  });
});

describe("upload row removal", () => {
  it("waits for the server while a file is being checked, so a refusal keeps the row", () => {
    expect(removalMode("uploaded_unverified")).toBe("confirm");
    expect(removalMode("verified")).toBe("release-verified");
    expect(removalMode("initiated")).toBe("best-effort");
    expect(removalMode("rejected")).toBe("best-effort");
    expect(removalMode(null)).toBe("best-effort");
  });
});
