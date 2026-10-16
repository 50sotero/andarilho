import { describe, expect, it } from "vitest";
import { applySimilarity, fitSimilarity2D, type LandmarkPair } from "../src/review/alignment";
import type { Vec2 } from "../src/types";

const truth = { scale: 0.02, rotationRadians: 0.7, translation: [3, -1.5] as Vec2 };
const planPoints: Vec2[] = [
  [0, 0],
  [400, 0],
  [400, 250],
  [0, 250],
  [120, 80],
];
const pairs: LandmarkPair[] = planPoints.map((plan) => ({
  plan,
  world: applySimilarity(truth, plan),
}));

describe("2D similarity fit", () => {
  it("recovers a known similarity exactly", () => {
    const result = fitSimilarity2D(pairs);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.fit.scale).toBeCloseTo(0.02, 10);
    expect(result.fit.rotationRadians).toBeCloseTo(0.7, 10);
    expect(result.fit.yawRadians).toBeCloseTo(-0.7, 10);
    expect(result.fit.translation[0]).toBeCloseTo(3, 9);
    expect(result.fit.translation[1]).toBeCloseTo(-1.5, 9);
    expect(result.fit.maxResidual).toBeLessThan(1e-9);
    expect(result.fit.disagreement).toBe(false);
    expect(result.fit.underdetermined).toBe(false);
  });

  it("reports residuals in meters and flags a disagreeing anchor", () => {
    const noisy = pairs.map((p, i) =>
      i === 4 ? { ...p, world: [p.world[0] + 0.5, p.world[1]] as Vec2 } : p,
    );
    const result = fitSimilarity2D(noisy, { tolerance: 0.1 });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.fit.residuals).toHaveLength(5);
    expect(result.fit.disagreement).toBe(true);
    expect(result.fit.disagreeingPairs).toContain(4);
    expect(result.fit.maxResidual).toBeGreaterThan(0.3);
    expect(result.fit.maxResidual).toBeLessThan(0.5);
  });

  it("initializes from two anchors but marks the fit underdetermined", () => {
    const result = fitSimilarity2D(pairs.slice(0, 2));
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.fit.underdetermined).toBe(true);
    expect(result.fit.scale).toBeCloseTo(0.02, 10);
  });

  it("rejects a mirrored plan instead of fitting a reflection", () => {
    const mirrored = pairs.map((p) => ({ ...p, plan: [p.plan[0], -p.plan[1]] as Vec2 }));
    const result = fitSimilarity2D(mirrored);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.reason).toBe("reflection");
  });

  it("rejects degenerate input", () => {
    expect(fitSimilarity2D(pairs.slice(0, 1))).toMatchObject({
      ok: false,
      reason: "too-few-pairs",
    });
    expect(
      fitSimilarity2D([
        { plan: [1, 1], world: [0, 0] },
        { plan: [1, 1], world: [2, 0] },
      ]),
    ).toMatchObject({ ok: false, reason: "degenerate-plan" });
    expect(
      fitSimilarity2D([
        { plan: [0, 0], world: [1, 1] },
        { plan: [1, 0], world: [1, 1] },
      ]),
    ).toMatchObject({ ok: false, reason: "degenerate-world" });
    expect(
      fitSimilarity2D([
        { plan: [0, Number.NaN], world: [0, 0] },
        { plan: [1, 0], world: [1, 0] },
      ]),
    ).toMatchObject({ ok: false, reason: "non-finite" });
  });

  it("flags collinear anchors", () => {
    const line = [0, 1, 2].map((i) => ({ plan: [i, 0] as Vec2, world: [i * 2, 1] as Vec2 }));
    const result = fitSimilarity2D(line);
    expect(result.ok && result.fit.collinear).toBe(true);
  });
});
