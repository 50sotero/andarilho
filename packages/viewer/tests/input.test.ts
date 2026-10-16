import { describe, expect, it } from "vitest";
import {
  DEFAULT_PITCH_LIMIT,
  applyLookDelta,
  clampPitch,
  keyCodeToAction,
  moveDirectionFromAxes,
  moveDirectionFromKeys,
  padAxes,
  wrapAngle,
  type MoveKeys,
} from "../src/movement/input";

const keys = (k: Partial<MoveKeys>): MoveKeys => ({
  forward: false,
  backward: false,
  left: false,
  right: false,
  ...k,
});

describe("keyboard mapping", () => {
  it("maps WASD and arrows", () => {
    expect(keyCodeToAction("KeyW")).toBe("forward");
    expect(keyCodeToAction("ArrowUp")).toBe("forward");
    expect(keyCodeToAction("KeyA")).toBe("left");
    expect(keyCodeToAction("ArrowRight")).toBe("right");
    expect(keyCodeToAction("KeyS")).toBe("backward");
    expect(keyCodeToAction("Space")).toBeNull();
  });
});

describe("move direction", () => {
  it("moves along camera forward (-Z) at yaw 0", () => {
    const [x, z] = moveDirectionFromKeys(keys({ forward: true }), 0);
    expect(x).toBeCloseTo(0);
    expect(z).toBeCloseTo(-1);
  });

  it("strafes right along +X at yaw 0", () => {
    const [x, z] = moveDirectionFromKeys(keys({ right: true }), 0);
    expect(x).toBeCloseTo(1);
    expect(z).toBeCloseTo(0);
  });

  it("is relative to yaw (yaw PI/2 looks down -X)", () => {
    const [x, z] = moveDirectionFromKeys(keys({ forward: true }), Math.PI / 2);
    expect(x).toBeCloseTo(-1);
    expect(z).toBeCloseTo(0);
    const [rx, rz] = moveDirectionFromKeys(keys({ right: true }), Math.PI / 2);
    expect(rx).toBeCloseTo(0);
    expect(rz).toBeCloseTo(-1);
  });

  it("does not move diagonally faster", () => {
    for (const yaw of [0, 0.3, 1.7, -2.4]) {
      const d = moveDirectionFromKeys(keys({ forward: true, right: true }), yaw);
      expect(Math.hypot(d[0], d[1])).toBeCloseTo(1, 12);
    }
    const analog = moveDirectionFromAxes({ x: 5, y: 5 }, 0);
    expect(Math.hypot(analog[0], analog[1])).toBeCloseTo(1, 12);
  });

  it("cancels opposing keys and keeps analog magnitude below one", () => {
    expect(moveDirectionFromKeys(keys({ forward: true, backward: true }), 1)).toEqual([0, 0]);
    const half = moveDirectionFromAxes({ x: 0, y: 0.5 }, 0);
    expect(Math.hypot(half[0], half[1])).toBeCloseTo(0.5);
  });
});

describe("look", () => {
  it("clamps pitch", () => {
    expect(clampPitch(10)).toBeCloseTo(DEFAULT_PITCH_LIMIT);
    expect(clampPitch(-10)).toBeCloseTo(-DEFAULT_PITCH_LIMIT);
    expect(clampPitch(0.2)).toBe(0.2);
    expect(clampPitch(Number.NaN)).toBe(0);
  });

  it("applies pointer deltas with bounded pitch", () => {
    let look = { yaw: 0, pitch: 0 };
    look = applyLookDelta(look, 100, 0, { sensitivity: 0.01 });
    expect(look.yaw).toBeCloseTo(-1); // moving right turns right
    look = applyLookDelta(look, 0, -10_000, { sensitivity: 0.01 });
    expect(look.pitch).toBeCloseTo(DEFAULT_PITCH_LIMIT); // moving up looks up, clamped
    look = applyLookDelta(look, 0, 10_000, { sensitivity: 0.01, pitchLimit: 0.5 });
    expect(look.pitch).toBeCloseTo(-0.5);
    const inverted = applyLookDelta({ yaw: 0, pitch: 0 }, 0, -10, {
      sensitivity: 0.01,
      invertY: true,
    });
    expect(inverted.pitch).toBeCloseTo(-0.1);
  });

  it("wraps yaw", () => {
    expect(wrapAngle(3 * Math.PI)).toBeCloseTo(Math.PI);
    expect(wrapAngle(-3 * Math.PI)).toBeCloseTo(Math.PI);
  });
});

describe("virtual move pad", () => {
  it("maps drag offset to unit-disc axes with a dead zone", () => {
    const up = padAxes(100, 100, 100, 40, 60);
    expect(up.x).toBeCloseTo(0);
    expect(up.y).toBeCloseTo(1);
    const far = padAxes(0, 0, 600, 600, 60);
    expect(Math.hypot(far.x, far.y)).toBeCloseTo(1);
    expect(padAxes(0, 0, 2, 2, 60)).toEqual({ x: 0, y: 0 });
  });
});
