import { readFileSync } from "node:fs";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  CollisionWorld,
  DEFAULT_CONTROLLER_CONFIG,
  initRapier,
  readCollisionGlb,
  type RapierModule,
} from "../src/movement/collision";
import { moveDirectionFromKeys } from "../src/movement/input";
import { DEFAULT_PLAYER_CONFIG } from "../src/movement/player";
import { layoutBounds, layoutToCollisionMesh } from "../src/review/floorplan";
import type { Aabb, CollisionMesh, Vec2, Vec3 } from "../src/types";
import { fixturePath, hasFixture, loadLayout } from "./helpers";

const STEP = 1 / 60;
const layout = loadLayout();
const spawn = layout.spawn.feet as Vec3;

function expand(b: Aabb, m: number): Aabb {
  return {
    min: [b.min[0] - m, b.min[1] - m, b.min[2] - m],
    max: [b.max[0] + m, b.max[1] + m, b.max[2] + m],
  };
}

function walk(world: CollisionWorld, direction: Vec2, seconds: number) {
  let last = { feet: world.state.feet, grounded: false, recovered: false };
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < Math.round(seconds / STEP); i++) {
    last = world.step(direction, STEP);
    minY = Math.min(minY, last.feet[1]);
    maxY = Math.max(maxY, last.feet[1]);
  }
  return { ...last, minY, maxY };
}

const towardMinusX = moveDirectionFromKeys(
  { forward: true, backward: false, left: false, right: false },
  layout.spawn.yawRadians,
);

function suite(name: string, getMesh: () => CollisionMesh) {
  describe(name, () => {
    let rapier: RapierModule;
    let world: CollisionWorld;

    beforeAll(async () => {
      rapier = await initRapier();
      world = new CollisionWorld(rapier, getMesh(), {
        player: { ...DEFAULT_PLAYER_CONFIG },
        bounds: expand(layoutBounds(layout), 0.2),
      });
      const v = world.validateSpawn(spawn);
      expect(v.ok).toBe(true);
      world.setSpawn(v.feet!);
    });

    afterAll(() => world?.dispose());

    it("uses the README controller settings", () => {
      expect(DEFAULT_CONTROLLER_CONFIG.offset).toBe(0.01);
      expect(DEFAULT_CONTROLLER_CONFIG.snapToGround).toBe(0.08);
      expect(DEFAULT_CONTROLLER_CONFIG.autostep).toBeNull();
      expect((DEFAULT_CONTROLLER_CONFIG.maxSlopeClimbAngle * 180) / Math.PI).toBeCloseTo(40);
    });

    it("validates the spawn with support and clearance", () => {
      const ok = world.validateSpawn(spawn);
      expect(ok.ok).toBe(true);
      expect(ok.supportY).toBeCloseTo(0, 3);
      expect(world.validateSpawn([-2, 0, -0.5])).toMatchObject({ ok: false, reason: "blocked" });
      expect(world.validateSpawn([0, 0, 1.5])).toMatchObject({ ok: false, reason: "blocked" });
      expect(world.validateSpawn([10, 0, 0])).toMatchObject({ ok: false, reason: "out-of-bounds" });
    });

    it("forward at the spawn yaw points toward the divider (-X)", () => {
      expect(towardMinusX[0]).toBeCloseTo(-1);
      expect(towardMinusX[1]).toBeCloseTo(0);
    });

    it("stays supported on the floor while idle (no falling)", () => {
      world.resetToSpawn();
      const r = walk(world, [0, 0], 2);
      expect(r.grounded).toBe(true);
      expect(r.feet[1]).toBeGreaterThan(-0.02);
      expect(r.feet[1]).toBeLessThan(0.05);
      expect(r.recovered).toBe(false);
    });

    it("is blocked by the divider wall away from the door", () => {
      world.teleport([2, 0, 1.5]);
      const r = walk(world, towardMinusX, 3); // would travel 4.2 m unobstructed
      const wallFace = 0.05 + DEFAULT_PLAYER_CONFIG.radius;
      expect(r.feet[0]).toBeGreaterThan(wallFace - 0.02);
      expect(r.feet[0]).toBeLessThan(wallFace + 0.1);
      expect(r.feet[2]).toBeCloseTo(1.5, 1);
      expect(r.minY).toBeGreaterThan(-0.02);
      expect(r.grounded).toBe(true);
    });

    it("walks through the doorway", () => {
      world.resetToSpawn();
      const r = walk(world, towardMinusX, 2); // 2.8 m: from x=2 to about x=-0.8
      expect(r.feet[0]).toBeLessThan(-0.5);
      expect(Math.abs(r.feet[2])).toBeLessThan(0.1);
      expect(r.minY).toBeGreaterThan(-0.02);
      expect(r.maxY).toBeLessThan(0.05);
      expect(r.grounded).toBe(true);
    });

    it("is blocked by the table", () => {
      world.teleport([-0.5, 0, -0.5]);
      const r = walk(world, [-1, 0], 3);
      const tableFace = -1.5 + DEFAULT_PLAYER_CONFIG.radius;
      expect(r.feet[0]).toBeGreaterThan(tableFace - 0.02);
      expect(r.feet[0]).toBeLessThan(tableFace + 0.1);
      expect(r.maxY).toBeLessThan(0.05); // did not climb onto the 0.75 m table
    });

    it("slides along walls instead of sticking", () => {
      world.teleport([2, 0, 1.5]);
      walk(world, [-1, 0], 2); // press into the divider
      const r = walk(world, [-Math.SQRT1_2, -Math.SQRT1_2], 1.5); // diagonal into the wall
      expect(r.feet[2]).toBeLessThan(1.0);
    });

    it("recovers to the last valid grounded position when leaving bounds", () => {
      world.resetToSpawn();
      walk(world, [0, 0], 0.5);
      const valid = [...world.state.lastValidFeet];
      world.teleport([2, -5, 0]);
      const r = world.step([0, 0], STEP);
      expect(r.recovered).toBe(true);
      expect(r.feet[0]).toBeCloseTo(valid[0]!, 3);
      expect(r.feet[2]).toBeCloseTo(valid[2]!, 3);
    });

    it("reset returns to spawn", () => {
      world.teleport([-3, 0, 1]);
      walk(world, [0, 1], 0.5);
      world.resetToSpawn();
      const feet = world.getFeet();
      expect(feet[0]).toBeCloseTo(spawn[0], 3);
      expect(feet[1]).toBeCloseTo(spawn[1], 2);
      expect(feet[2]).toBeCloseTo(spawn[2], 3);
      const r = walk(world, [0, 0], 0.5);
      expect(r.feet[0]).toBeCloseTo(spawn[0], 3);
      expect(r.grounded).toBe(true);
    });
  });
}

suite("walking on layout-generated collision (Rapier in Node)", () =>
  layoutToCollisionMesh(layout),
);

describe.skipIf(!hasFixture("synthetic-two-room.collision.glb"))("packaged fixture GLB", () => {
  suite(
    "walking on the fixture collision GLB",
    () => readCollisionGlb(readFileSync(fixturePath("synthetic-two-room.collision.glb"))).mesh,
  );
});

describe("CollisionWorld lifecycle", () => {
  it("frees the Rapier world once and rejects use after dispose", async () => {
    const rapier = await initRapier();
    const world = new CollisionWorld(rapier, layoutToCollisionMesh(layout), {
      player: { ...DEFAULT_PLAYER_CONFIG },
    });
    world.dispose();
    world.dispose();
    expect(world.disposed).toBe(true);
    expect(() => world.step([0, 0], STEP)).toThrow(/disposed/);
  });
});
