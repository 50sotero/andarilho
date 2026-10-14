import type RAPIER_NS from "@dimforge/rapier3d-compat";
import type {
  Collider,
  KinematicCharacterController,
  RigidBody,
  World,
} from "@dimforge/rapier3d-compat";
import type { Aabb, CollisionMesh, Vec2, Vec3 } from "../types";
import { meshBounds } from "../types";
import {
  capsuleCenterY,
  capsuleHalfHeight,
  desiredTranslation,
  feetYFromCenterY,
  integrateVertical,
  PlayerState,
  type PlayerConfig,
} from "./player";

export {
  CollisionAssetError,
  collisionMeshIssues,
  nodeTransformIssue,
  parseGlb,
  readCollisionGlb,
  validateCollisionMesh,
  writeCollisionGlb,
  type CollisionMeshStats,
  type CollisionValidationOptions,
  type ValidatedCollisionMesh,
} from "./glb";

export type RapierModule = typeof RAPIER_NS;

let rapierPromise: Promise<RapierModule> | null = null;

/**
 * Initializes the Rapier compat build (WASM inlined as base64) exactly once.
 * Works in browsers and in Node (tests). A failed init can be retried.
 */
export function initRapier(): Promise<RapierModule> {
  if (!rapierPromise) {
    rapierPromise = import("@dimforge/rapier3d-compat")
      .then(async (mod) => {
        const rapier = (mod.default ?? mod) as RapierModule;
        await rapier.init();
        return rapier;
      })
      .catch((error: unknown) => {
        rapierPromise = null;
        throw error;
      });
  }
  return rapierPromise;
}

/** Character controller settings (README section 11). */
export interface ControllerConfig {
  /** Rapier controller skin offset in meters. */
  offset: number;
  /** Maximum climbable slope in radians (~40 degrees). */
  maxSlopeClimbAngle: number;
  /** Slopes steeper than this slide in radians. */
  minSlopeSlideAngle: number;
  /** Snap-to-ground distance in meters. */
  snapToGround: number;
  /** Autostep stays disabled for the first single-floor release. */
  autostep: null | { maxHeight: number; minWidth: number };
}

export const DEFAULT_CONTROLLER_CONFIG: Readonly<ControllerConfig> = Object.freeze({
  offset: 0.01,
  maxSlopeClimbAngle: (40 * Math.PI) / 180,
  minSlopeSlideAngle: (40 * Math.PI) / 180,
  snapToGround: 0.08,
  autostep: null,
});

export interface FixedStepOptions {
  /** Fixed physics step in seconds. */
  step?: number;
  /** Maximum accumulated frame delta in seconds. */
  maxFrameDelta?: number;
  /** Maximum fixed substeps per rendered frame. */
  maxSubsteps?: number;
}

export interface FixedStepResult {
  /** Number of fixed steps to run this frame. */
  steps: number;
  /** Interpolation factor between the previous and current physics state. */
  alpha: number;
  /** Frame delta actually accepted after clamping. */
  acceptedDelta: number;
  /** Simulation time discarded because of clamping or the substep bound. */
  droppedTime: number;
}

/**
 * Fixed-step accumulator decoupling physics from rendering. A long pause (for
 * example a background tab) is clamped to `maxFrameDelta` so it never produces
 * one huge movement step, and substeps per frame are bounded.
 */
export class FixedStepAccumulator {
  readonly step: number;
  readonly maxFrameDelta: number;
  readonly maxSubsteps: number;
  private accumulator = 0;

  constructor(options: FixedStepOptions = {}) {
    this.step = options.step ?? 1 / 60;
    this.maxFrameDelta = options.maxFrameDelta ?? 0.1;
    this.maxSubsteps = options.maxSubsteps ?? 6;
    if (!(this.step > 0) || !(this.maxFrameDelta > 0) || !(this.maxSubsteps >= 1)) {
      throw new RangeError("Invalid fixed-step configuration");
    }
  }

  advance(frameDelta: number): FixedStepResult {
    let dropped = 0;
    let delta = Number.isFinite(frameDelta) && frameDelta > 0 ? frameDelta : 0;
    if (delta > this.maxFrameDelta) {
      dropped += delta - this.maxFrameDelta;
      delta = this.maxFrameDelta;
    }
    this.accumulator += delta;
    let steps = Math.floor(this.accumulator / this.step + 1e-9);
    if (steps > this.maxSubsteps) {
      dropped += (steps - this.maxSubsteps) * this.step;
      steps = this.maxSubsteps;
      this.accumulator = Math.min(this.accumulator - steps * this.step, this.step * 0.999);
    } else {
      this.accumulator -= steps * this.step;
    }
    if (this.accumulator < 0) this.accumulator = 0;
    return {
      steps,
      alpha: this.accumulator / this.step,
      acceptedDelta: delta,
      droppedTime: dropped,
    };
  }

  reset(): void {
    this.accumulator = 0;
  }
}

export interface SpawnValidation {
  ok: boolean;
  reason?: "no-support" | "blocked" | "out-of-bounds" | "invalid";
  /** Y of the supporting surface found below the spawn, if any. */
  supportY?: number;
  /** Feet position adjusted onto the support surface. */
  feet?: Vec3;
}

export interface CollisionWorldOptions {
  player: PlayerConfig;
  controller?: Partial<ControllerConfig>;
  /** Approved bounds for out-of-bounds recovery; defaults to the mesh bounds. */
  bounds?: Aabb | null;
  /** Fixed step used by the physics world. */
  step?: number;
  /** Maximum distance below the spawn feet at which support must exist. */
  maxSupportDistance?: number;
}

export interface StepResult {
  feet: Vec3;
  grounded: boolean;
  recovered: boolean;
}

/**
 * Rapier world with static trimesh colliders from world-space collision
 * geometry and a kinematic position-based capsule driven by a
 * KinematicCharacterController (README section 11, steps 3-8).
 */
export class CollisionWorld {
  readonly rapier: RapierModule;
  readonly player: PlayerConfig;
  readonly controllerConfig: ControllerConfig;
  readonly bounds: Aabb | null;
  readonly state: PlayerState;
  private world: World | null;
  private readonly staticBody: RigidBody;
  private readonly body: RigidBody;
  private readonly capsule: Collider;
  private readonly controller: KinematicCharacterController;
  private readonly maxSupportDistance: number;
  private spawnFeet: Vec3 = [0, 0, 0];

  constructor(rapier: RapierModule, mesh: CollisionMesh, options: CollisionWorldOptions) {
    this.rapier = rapier;
    this.player = options.player;
    this.controllerConfig = { ...DEFAULT_CONTROLLER_CONFIG, ...options.controller };
    this.bounds = options.bounds ?? meshBounds(mesh);
    this.maxSupportDistance = options.maxSupportDistance ?? 0.5;

    const world = new rapier.World({ x: 0, y: 0, z: 0 });
    this.world = world;
    try {
      world.timestep = options.step ?? 1 / 60;
      this.staticBody = world.createRigidBody(rapier.RigidBodyDesc.fixed());
      const flags = rapier.TriMeshFlags?.FIX_INTERNAL_EDGES;
      const trimesh =
        flags !== undefined
          ? rapier.ColliderDesc.trimesh(mesh.positions, mesh.indices, flags)
          : rapier.ColliderDesc.trimesh(mesh.positions, mesh.indices);
      world.createCollider(trimesh, this.staticBody);

      this.body = world.createRigidBody(
        rapier.RigidBodyDesc.kinematicPositionBased().setTranslation(0, -1000, 0),
      );
      this.capsule = world.createCollider(
        rapier.ColliderDesc.capsule(capsuleHalfHeight(this.player), this.player.radius),
        this.body,
      );

      const c = this.controllerConfig;
      this.controller = world.createCharacterController(c.offset);
      this.controller.setUp({ x: 0, y: 1, z: 0 });
      this.controller.setSlideEnabled(true);
      this.controller.setMaxSlopeClimbAngle(c.maxSlopeClimbAngle);
      this.controller.setMinSlopeSlideAngle(c.minSlopeSlideAngle);
      this.controller.enableSnapToGround(c.snapToGround);
      if (c.autostep)
        this.controller.enableAutostep(c.autostep.maxHeight, c.autostep.minWidth, false);
      else this.controller.disableAutostep();
      this.controller.setApplyImpulsesToDynamicBodies(false);
      world.step();
    } catch (error) {
      world.free();
      this.world = null;
      throw error;
    }
    this.state = new PlayerState([0, 0, 0]);
  }

  get disposed(): boolean {
    return this.world === null;
  }

  private requireWorld(): World {
    if (!this.world) throw new Error("CollisionWorld has been disposed");
    return this.world;
  }

  /**
   * Validates a spawn point: a downward ray from inside the capsule must hit a
   * support surface within `maxSupportDistance` below the feet, and a capsule
   * placed on that surface (plus the controller offset) must not intersect
   * any collider.
   */
  validateSpawn(feet: Vec3): SpawnValidation {
    const world = this.requireWorld();
    if (!feet.every((v) => Number.isFinite(v))) return { ok: false, reason: "invalid" };
    if (this.bounds) {
      for (let k = 0; k < 3; k++) {
        if (feet[k]! < this.bounds.min[k]! - 0.05 || feet[k]! > this.bounds.max[k]! + 0.05) {
          return { ok: false, reason: "out-of-bounds" };
        }
      }
    }
    const r = this.rapier;
    const originY = feet[1] + this.player.radius;
    const ray = new r.Ray({ x: feet[0], y: originY, z: feet[2] }, { x: 0, y: -1, z: 0 });
    const maxToi = this.player.radius + this.maxSupportDistance;
    const hit = world.castRay(ray, maxToi, true, undefined, undefined, this.capsule, this.body);
    if (!hit) return { ok: false, reason: "no-support" };
    const supportY = originY - hit.timeOfImpact;
    const placed: Vec3 = [feet[0], supportY, feet[2]];
    const shape = new r.Capsule(capsuleHalfHeight(this.player), this.player.radius);
    const centerY = capsuleCenterY(supportY, this.player) + this.controllerConfig.offset * 2;
    const blocked = world.intersectionWithShape(
      { x: feet[0], y: centerY, z: feet[2] },
      { x: 0, y: 0, z: 0, w: 1 },
      shape,
      undefined,
      undefined,
      this.capsule,
      this.body,
    );
    if (blocked) return { ok: false, reason: "blocked", supportY };
    return { ok: true, supportY, feet: placed };
  }

  /** Places the capsule at a feet position (spawn/reset/recovery). */
  teleport(feet: Vec3): void {
    const world = this.requireWorld();
    const centerY = capsuleCenterY(feet[1], this.player) + this.controllerConfig.offset;
    this.body.setTranslation({ x: feet[0], y: centerY, z: feet[2] }, true);
    this.body.setNextKinematicTranslation({ x: feet[0], y: centerY, z: feet[2] });
    world.step();
    this.state.resetTo(feet);
  }

  /** Sets the spawn (after validation) and teleports to it. */
  setSpawn(feet: Vec3): void {
    this.spawnFeet = [...feet];
    this.state.lastValidFeet = [...feet];
    this.teleport(feet);
  }

  getSpawn(): Vec3 {
    return [...this.spawnFeet];
  }

  resetToSpawn(): void {
    this.teleport(this.spawnFeet);
    this.state.lastValidFeet = [...this.spawnFeet];
  }

  /** Current feet position derived from the body center. */
  getFeet(): Vec3 {
    const t = this.body.translation();
    return [t.x, feetYFromCenterY(t.y - this.controllerConfig.offset, this.player), t.z];
  }

  /**
   * One fixed step: integrate gravity, ask the character controller to correct
   * the requested translation, move the kinematic body, step the world, reset
   * downward velocity when grounded, and recover to the last valid grounded
   * position when the player leaves the approved bounds.
   */
  step(direction: Vec2, dt: number): StepResult {
    const world = this.requireWorld();
    const state = this.state;
    const vertical = integrateVertical(state.verticalVelocity, state.grounded, dt, this.player);
    state.verticalVelocity = vertical.velocity;
    const desired = desiredTranslation(direction, vertical.displacement, dt, this.player);

    this.controller.computeColliderMovement(
      this.capsule,
      { x: desired[0], y: desired[1], z: desired[2] },
      this.rapier.QueryFilterFlags.EXCLUDE_SENSORS,
    );
    const corrected = this.controller.computedMovement();
    const grounded = this.controller.computedGrounded();
    const current = this.body.translation();
    this.body.setNextKinematicTranslation({
      x: current.x + corrected.x,
      y: current.y + corrected.y,
      z: current.z + corrected.z,
    });
    world.step();

    const feet = this.getFeet();
    const needsRecovery = state.commit(feet, grounded, this.bounds);
    if (needsRecovery) {
      const target = state.lastValidFeet;
      this.teleport(target);
      state.recoveries++;
      return { feet: [...target], grounded: false, recovered: true };
    }
    return { feet, grounded, recovered: false };
  }

  /** Frees every Rapier object owned by this world. Safe to call twice. */
  dispose(): void {
    if (!this.world) return;
    const world = this.world;
    this.world = null;
    try {
      world.removeCharacterController(this.controller);
    } catch {
      // The world is freed below regardless.
    }
    world.free();
  }
}
