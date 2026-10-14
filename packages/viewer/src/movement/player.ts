import type { Aabb, Vec2, Vec3 } from "../types";

/**
 * Tunable product defaults for the walking capsule (README section 11). They are
 * not claims about human dimensions. The worker's spawn/clearance validation
 * must use compatible values.
 */
export interface PlayerConfig {
  /** Capsule radius in meters. */
  radius: number;
  /** Total capsule height (feet to top of head) in meters. */
  height: number;
  /** Eye height above the feet in meters. */
  eyeHeight: number;
  /** Horizontal walking speed in meters per second. */
  walkSpeed: number;
  /** Gravity acceleration magnitude in m/s^2 (application-managed). */
  gravity: number;
  /** Terminal falling speed in m/s. */
  maxFallSpeed: number;
}

export const DEFAULT_PLAYER_CONFIG: Readonly<PlayerConfig> = Object.freeze({
  radius: 0.25,
  height: 1.7,
  eyeHeight: 1.6,
  walkSpeed: 1.4,
  gravity: 9.81,
  maxFallSpeed: 20,
});

export function resolvePlayerConfig(overrides?: Partial<PlayerConfig>): PlayerConfig {
  const config: PlayerConfig = { ...DEFAULT_PLAYER_CONFIG };
  if (overrides) {
    for (const key of Object.keys(config) as (keyof PlayerConfig)[]) {
      const value = overrides[key];
      if (typeof value === "number" && Number.isFinite(value)) config[key] = value;
    }
  }
  const errors = validatePlayerConfig(config);
  if (errors.length > 0) throw new RangeError(`Invalid player config: ${errors.join("; ")}`);
  return config;
}

export function validatePlayerConfig(config: PlayerConfig): string[] {
  const errors: string[] = [];
  if (!(config.radius > 0)) errors.push("radius must be positive");
  if (!(config.height > 2 * config.radius)) errors.push("height must exceed twice the radius");
  if (!(config.eyeHeight > 0 && config.eyeHeight <= config.height)) {
    errors.push("eyeHeight must be within (0, height]");
  }
  if (!(config.walkSpeed > 0)) errors.push("walkSpeed must be positive");
  if (!(config.gravity >= 0)) errors.push("gravity must be non-negative");
  if (!(config.maxFallSpeed > 0)) errors.push("maxFallSpeed must be positive");
  return errors;
}

/** Cylinder half-height of the capsule: (totalHeight - 2 * radius) / 2. */
export function capsuleHalfHeight(config: Pick<PlayerConfig, "height" | "radius">): number {
  return (config.height - 2 * config.radius) / 2;
}

/** Capsule body center Y for a given feet Y: feetY + totalHeight / 2. */
export function capsuleCenterY(feetY: number, config: Pick<PlayerConfig, "height">): number {
  return feetY + config.height / 2;
}

/** Inverse of {@link capsuleCenterY}. */
export function feetYFromCenterY(centerY: number, config: Pick<PlayerConfig, "height">): number {
  return centerY - config.height / 2;
}

/** Eye Y for a given feet Y. */
export function eyeYFromFeetY(feetY: number, config: Pick<PlayerConfig, "eyeHeight">): number {
  return feetY + config.eyeHeight;
}

export function feetToCenter(feet: Vec3, config: Pick<PlayerConfig, "height">): Vec3 {
  return [feet[0], capsuleCenterY(feet[1], config), feet[2]];
}

export function centerToFeet(center: Vec3, config: Pick<PlayerConfig, "height">): Vec3 {
  return [center[0], feetYFromCenterY(center[1], config), center[2]];
}

export function feetToEye(feet: Vec3, config: Pick<PlayerConfig, "eyeHeight">): Vec3 {
  return [feet[0], eyeYFromFeetY(feet[1], config), feet[2]];
}

export interface VerticalStep {
  /** Vertical velocity after integration (m/s, negative is down). */
  velocity: number;
  /** Requested vertical displacement for this step (m). */
  displacement: number;
}

/**
 * Semi-implicit Euler integration of application-managed gravity. When the
 * controller reported the character grounded on the previous step, downward
 * velocity is reset first so falling speed never accumulates on the floor.
 */
export function integrateVertical(
  velocity: number,
  grounded: boolean,
  dt: number,
  config: Pick<PlayerConfig, "gravity" | "maxFallSpeed">,
): VerticalStep {
  let v = grounded && velocity < 0 ? 0 : velocity;
  v -= config.gravity * dt;
  if (v < -config.maxFallSpeed) v = -config.maxFallSpeed;
  return { velocity: v, displacement: v * dt };
}

/**
 * Builds the desired translation for one fixed step from a normalized XZ
 * direction (length <= 1) and the integrated vertical displacement.
 */
export function desiredTranslation(
  direction: Vec2,
  verticalDisplacement: number,
  dt: number,
  config: Pick<PlayerConfig, "walkSpeed">,
): Vec3 {
  const len = Math.hypot(direction[0], direction[1]);
  const scale = len > 1 ? 1 / len : 1;
  return [
    direction[0] * scale * config.walkSpeed * dt,
    verticalDisplacement,
    direction[1] * scale * config.walkSpeed * dt,
  ];
}

/** True when the feet point is outside the approved bounds (with a margin). */
export function isOutOfBounds(feet: Vec3, bounds: Aabb, margin = 0.5): boolean {
  for (let k = 0; k < 3; k++) {
    const v = feet[k] as number;
    if (!Number.isFinite(v)) return true;
    if (v < (bounds.min[k] as number) - margin || v > (bounds.max[k] as number) + margin) {
      return true;
    }
  }
  return false;
}

/**
 * Mutable per-player navigation state kept by the engine (never in React
 * state). Tracks the last valid grounded position used for recovery.
 */
export class PlayerState {
  feet: Vec3;
  previousFeet: Vec3;
  verticalVelocity = 0;
  grounded = false;
  lastValidFeet: Vec3;
  recoveries = 0;

  constructor(spawnFeet: Vec3) {
    this.feet = [...spawnFeet];
    this.previousFeet = [...spawnFeet];
    this.lastValidFeet = [...spawnFeet];
  }

  /** Records the result of a physics step and returns true when recovery is needed. */
  commit(feet: Vec3, grounded: boolean, bounds: Aabb | null, margin = 0.5): boolean {
    this.previousFeet = this.feet;
    this.feet = [...feet];
    this.grounded = grounded;
    if (grounded && this.verticalVelocity < 0) this.verticalVelocity = 0;
    const outside = bounds ? isOutOfBounds(this.feet, bounds, margin) : false;
    if (outside) return true;
    if (grounded) this.lastValidFeet = [...this.feet];
    return false;
  }

  /** Teleports to a position (spawn or last valid), clearing vertical motion. */
  resetTo(feet: Vec3): void {
    this.feet = [...feet];
    this.previousFeet = [...feet];
    this.verticalVelocity = 0;
    this.grounded = false;
  }

  /** Interpolated feet between the previous and current physics step. */
  interpolatedFeet(alpha: number): Vec3 {
    const a = Math.min(1, Math.max(0, alpha));
    const p = this.previousFeet;
    const c = this.feet;
    return [p[0] + (c[0] - p[0]) * a, p[1] + (c[1] - p[1]) * a, p[2] + (c[2] - p[2]) * a];
  }
}
