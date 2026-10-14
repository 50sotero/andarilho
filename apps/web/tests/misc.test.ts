import { describe, expect, it } from "vitest";
import { safeNextPath } from "@/lib/auth/safe-redirect";
import {
  calibrationFactor,
  isSelfIntersecting,
  layoutProblems,
  scaleLayout,
} from "@/features/review/geometry";
import { addOpening, movePoint, removeSelected } from "@/features/review/layout-edit";
import { formatStageProgress, stageStates } from "@/features/processing/stages";
import { expiryFromDays } from "@/features/sharing/SharePanel";
import type { Layout, Run } from "@/lib/api/types";
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
    expect(scaled.walls![0]!.height).toBe(layout.walls![0]!.height);
  });
  it("adds openings inside walls and removes a wall's openings with it", () => {
    const withOpening = addOpening(layout, 0);
    expect(layoutProblems(withOpening)).toEqual([]);
    const removed = removeSelected(withOpening, { kind: "wall", index: 0 });
    expect(removed.openings!.some((o) => o.wallId === layout.walls![0]!.id)).toBe(false);
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
