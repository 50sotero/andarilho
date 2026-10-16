import { describe, expect, it } from "vitest";
import {
  alignPlan,
  describeAlignment,
  landmarkPairs,
  setPlanPoint,
  underlayFromFit,
} from "@/features/review/alignment";
import type { PlanUnderlay } from "@/features/review/LayoutCanvas";
import type { Layout } from "@/lib/api/types";
import fixture from "@andarilho/scene-contract/fixtures/synthetic-two-room.layout.json";

type Vec2 = [number, number];

// A plan image 12 m wide, turned 30 degrees and offset against the capture.
const WIDTH = 12;
const ANGLE = Math.PI / 6;
const OFFSET: Vec2 = [2, -1];

function planToWorld([u, v]: Vec2): Vec2 {
  const c = Math.cos(ANGLE);
  const s = Math.sin(ANGLE);
  return [OFFSET[0] + WIDTH * (c * u - s * v), OFFSET[1] + WIDTH * (s * u + c * v)];
}

function withLandmarks(points: { plan: Vec2 | null; world: Vec2 }[]): Layout {
  const layout = structuredClone(fixture as unknown as Layout);
  layout.landmarks = points.map((p, i) => ({
    id: `landmark-${i + 1}`,
    planPoint: p.plan,
    worldPoint: [p.world[0], 0, p.world[1]],
    uncertainty: 0.05,
  }));
  return layout;
}

const PLAN: Vec2[] = [
  [0.1, 0.1],
  [0.8, 0.15],
  [0.3, 0.6],
  [0.75, 0.55],
];

/** Where the editor's SVG draws a plan point: rotate(deg, x, z) about the image's corner. */
function drawn(underlay: PlanUnderlay, [u, v]: Vec2): Vec2 {
  const a = ((underlay.rotationDegrees ?? 0) * Math.PI) / 180;
  const px = underlay.width * u;
  const pz = underlay.width * v;
  return [
    underlay.x + Math.cos(a) * px - Math.sin(a) * pz,
    underlay.z + Math.sin(a) * px + Math.cos(a) * pz,
  ];
}

describe("plan alignment", () => {
  it("pairs only landmarks marked on the plan, on the floor plane", () => {
    const layout = withLandmarks([
      { plan: [0.1, 0.2], world: [1, 2] },
      { plan: null, world: [3, 4] },
    ]);
    expect(landmarkPairs(layout)).toEqual([
      { id: "landmark-1", plan: [0.1, 0.2], world: [1, 2], uncertainty: 0.05 },
    ]);
    expect(alignPlan(layout)).toBeNull();
  });

  it("recovers the plan's width, turn and offset and places the underlay on the landmarks", () => {
    const layout = withLandmarks(PLAN.map((plan) => ({ plan, world: planToWorld(plan) })));
    const alignment = alignPlan(layout);
    expect(alignment?.result.ok).toBe(true);
    if (!alignment?.result.ok) return;
    const fit = alignment.result.fit;
    expect(fit.scale).toBeCloseTo(WIDTH, 6);
    expect(fit.rotationRadians).toBeCloseTo(ANGLE, 6);
    expect(fit.maxResidual).toBeLessThan(1e-6);
    const underlay = underlayFromFit(fit, "https://plan.example/plan.png", 0.5);
    expect(underlay).toMatchObject({ x: 2, z: -1, width: 12, rotationDegrees: 30 });
    for (const plan of PLAN) {
      const [x, z] = drawn(underlay, plan);
      const [wx, wz] = planToWorld(plan);
      expect(x).toBeCloseTo(wx, 2);
      expect(z).toBeCloseTo(wz, 2);
    }
    const { summary, warnings } = describeAlignment(alignment);
    expect(summary[0]).toContain("12.00 m wide");
    expect(warnings).toEqual([]);
  });

  it("names a landmark that disagrees with the plan", () => {
    const points = PLAN.map((plan) => ({ plan, world: planToWorld(plan) }));
    points[2]!.world = [points[2]!.world[0] + 0.3, points[2]!.world[1]]; // mis-marked by 30 cm
    const alignment = alignPlan(withLandmarks(points));
    expect(alignment?.result.ok).toBe(true);
    const { warnings } = describeAlignment(alignment!);
    expect(alignment?.suspect?.id).toBe("landmark-3");
    expect(alignment?.suspect?.residual).toBeCloseTo(0.3, 6);
    expect(warnings.join(" ")).toMatch(/landmark-3 looks misplaced.*0\.30 m/);
  });

  it("names every disagreeing landmark when no single one explains it", () => {
    const points = PLAN.slice(0, 3).map((plan) => ({ plan, world: planToWorld(plan) }));
    points[2]!.world = [points[2]!.world[0] + 0.5, points[2]!.world[1]];
    const alignment = alignPlan(withLandmarks(points))!;
    expect(alignment.suspect).toBeNull(); // three landmarks: any two fit exactly
    expect(describeAlignment(alignment).warnings.join(" ")).toMatch(/are more than 0\.10 m/);
  });

  it("warns that two landmarks cannot check the plan", () => {
    const points = PLAN.slice(0, 2).map((plan) => ({ plan, world: planToWorld(plan) }));
    const { warnings } = describeAlignment(alignPlan(withLandmarks(points))!);
    expect(warnings.join(" ")).toMatch(/Two landmarks place the plan but cannot check it/);
  });

  it("refuses a mirror-image fit", () => {
    const points = PLAN.map((plan) => {
      const [x, z] = planToWorld(plan);
      return { plan, world: [x, -z] as Vec2 };
    });
    const alignment = alignPlan(withLandmarks(points));
    expect(alignment?.result.ok).toBe(false);
    expect(describeAlignment(alignment!).warnings.join(" ")).toMatch(/mirror image/);
  });

  it("sets and clears a landmark's plan mark without touching the others", () => {
    const layout = withLandmarks([
      { plan: null, world: [1, 2] },
      { plan: [0.5, 0.5], world: [3, 4] },
    ]);
    const marked = setPlanPoint(layout, 0, [0.123456, 0.654321]);
    expect(marked.landmarks?.[0]?.planPoint).toEqual([0.1235, 0.6543]);
    expect(marked.landmarks?.[1]).toBe(layout.landmarks?.[1]);
    expect(setPlanPoint(marked, 0, null).landmarks?.[0]?.planPoint).toBeNull();
    expect(setPlanPoint(layout, 5, [0, 0])).toBe(layout);
  });
});
