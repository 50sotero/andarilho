import { describe, expect, it } from "vitest";
import {
  DEFAULT_PLAYER_CONFIG,
  PlayerState,
  capsuleCenterY,
  capsuleHalfHeight,
  centerToFeet,
  desiredTranslation,
  eyeYFromFeetY,
  feetToCenter,
  integrateVertical,
  isOutOfBounds,
  resolvePlayerConfig,
} from "../src/movement/player";
import type { Aabb, Vec3 } from "../src/types";

describe("capsule math", () => {
  it("uses the README defaults", () => {
    expect(DEFAULT_PLAYER_CONFIG.radius).toBe(0.25);
    expect(DEFAULT_PLAYER_CONFIG.height).toBe(1.7);
    expect(DEFAULT_PLAYER_CONFIG.eyeHeight).toBe(1.6);
    expect(DEFAULT_PLAYER_CONFIG.walkSpeed).toBe(1.4);
  });

  it("computes half-height, center and eye from feet", () => {
    const cfg = DEFAULT_PLAYER_CONFIG;
    expect(capsuleHalfHeight(cfg)).toBeCloseTo((1.7 - 0.5) / 2, 12);
    expect(capsuleCenterY(0, cfg)).toBeCloseTo(0.85, 12);
    expect(capsuleCenterY(1.2, cfg)).toBeCloseTo(2.05, 12);
    expect(eyeYFromFeetY(0.3, cfg)).toBeCloseTo(1.9, 12);
    // The capsule bottom touches the feet and its top touches the head.
    const center = capsuleCenterY(0, cfg);
    expect(center - capsuleHalfHeight(cfg) - cfg.radius).toBeCloseTo(0, 12);
    expect(center + capsuleHalfHeight(cfg) + cfg.radius).toBeCloseTo(1.7, 12);
  });

  it("round-trips feet and center", () => {
    const feet: Vec3 = [1, 0.5, -2];
    const back = centerToFeet(feetToCenter(feet, DEFAULT_PLAYER_CONFIG), DEFAULT_PLAYER_CONFIG);
    back.forEach((v, i) => expect(v).toBeCloseTo(feet[i]!, 12));
  });

  it("rejects impossible configs", () => {
    expect(() => resolvePlayerConfig({ radius: 1, height: 1.5 })).toThrow(RangeError);
    expect(() => resolvePlayerConfig({ eyeHeight: 2 })).toThrow(RangeError);
    expect(resolvePlayerConfig({ radius: Number.NaN }).radius).toBe(0.25);
  });
});

describe("vertical integration", () => {
  it("accumulates gravity while airborne and clamps terminal speed", () => {
    let v = 0;
    for (let i = 0; i < 60; i++)
      v = integrateVertical(v, false, 1 / 60, DEFAULT_PLAYER_CONFIG).velocity;
    expect(v).toBeCloseTo(-9.81, 6);
    expect(integrateVertical(-100, false, 1 / 60, DEFAULT_PLAYER_CONFIG).velocity).toBe(-20);
  });

  it("resets downward velocity when grounded", () => {
    const step = integrateVertical(-5, true, 1 / 60, DEFAULT_PLAYER_CONFIG);
    expect(step.velocity).toBeCloseTo(-9.81 / 60, 9);
    expect(step.displacement).toBeCloseTo(-9.81 / 3600, 9);
  });

  it("builds a speed-limited translation", () => {
    const t = desiredTranslation([3, 4], -0.01, 0.5, DEFAULT_PLAYER_CONFIG);
    expect(Math.hypot(t[0], t[2])).toBeCloseTo(0.7, 9);
    expect(t[1]).toBe(-0.01);
  });
});

describe("PlayerState", () => {
  const bounds: Aabb = { min: [-1, -1, -1], max: [1, 3, 1] };

  it("keeps the last valid grounded position and flags out-of-bounds", () => {
    const s = new PlayerState([0, 0, 0]);
    expect(s.commit([0.5, 0, 0], true, bounds)).toBe(false);
    expect(s.lastValidFeet).toEqual([0.5, 0, 0]);
    expect(s.commit([0.6, 0.2, 0], false, bounds)).toBe(false);
    expect(s.lastValidFeet).toEqual([0.5, 0, 0]);
    expect(s.commit([5, 0, 0], true, bounds)).toBe(true);
    expect(s.lastValidFeet).toEqual([0.5, 0, 0]);
  });

  it("detects non-finite positions as out of bounds", () => {
    expect(isOutOfBounds([Number.NaN, 0, 0], bounds)).toBe(true);
    expect(isOutOfBounds([0, -5, 0], bounds)).toBe(true);
    expect(isOutOfBounds([1.2, 0, 0], bounds, 0.5)).toBe(false);
  });

  it("interpolates between physics steps", () => {
    const s = new PlayerState([0, 0, 0]);
    s.commit([1, 0, 0], true, null);
    expect(s.interpolatedFeet(0.25)[0]).toBeCloseTo(0.25);
  });
});
