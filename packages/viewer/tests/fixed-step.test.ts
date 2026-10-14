import { describe, expect, it } from "vitest";
import { FixedStepAccumulator } from "../src/movement/collision";

describe("FixedStepAccumulator", () => {
  it("runs one step per 1/60 s frame", () => {
    const acc = new FixedStepAccumulator();
    let steps = 0;
    for (let i = 0; i < 60; i++) steps += acc.advance(1 / 60).steps;
    expect(steps).toBe(60);
  });

  it("clamps a long pause to the max frame delta", () => {
    const acc = new FixedStepAccumulator({ step: 1 / 60, maxFrameDelta: 0.1, maxSubsteps: 10 });
    const r = acc.advance(30); // e.g. returning from a background tab
    expect(r.acceptedDelta).toBeCloseTo(0.1);
    expect(r.steps).toBe(6);
    expect(r.droppedTime).toBeCloseTo(29.9);
    // Nothing large is carried over into the next frame.
    expect(acc.advance(0).steps).toBe(0);
  });

  it("bounds substeps per frame", () => {
    const acc = new FixedStepAccumulator({ step: 1 / 120, maxFrameDelta: 0.1, maxSubsteps: 4 });
    const r = acc.advance(0.1);
    expect(r.steps).toBe(4);
    expect(r.alpha).toBeLessThan(1);
    expect(acc.advance(0).steps).toBe(0);
  });

  it("ignores invalid deltas and resets", () => {
    const acc = new FixedStepAccumulator();
    expect(acc.advance(Number.NaN).steps).toBe(0);
    expect(acc.advance(-1).steps).toBe(0);
    acc.advance(0.01);
    acc.reset();
    expect(acc.advance(0.01).steps).toBe(0);
  });

  it("keeps the interpolation alpha in [0, 1)", () => {
    const acc = new FixedStepAccumulator();
    for (const dt of [0.005, 0.013, 0.021, 0.04, 0.0999]) {
      const { alpha } = acc.advance(dt);
      expect(alpha).toBeGreaterThanOrEqual(0);
      expect(alpha).toBeLessThan(1);
    }
  });
});
