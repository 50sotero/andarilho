/** Shared plain-data types used by the pure (Node-testable) viewer modules. */

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];

export interface Aabb {
  min: Vec3;
  max: Vec3;
}

/**
 * World-space triangle soup. `positions` holds xyz triplets in the canonical
 * Andarilho frame (meters, +Y up); `indices` holds triangle vertex indices.
 */
export interface CollisionMesh {
  positions: Float32Array;
  indices: Uint32Array;
}

export function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function isFiniteVec(values: readonly unknown[], length: number): boolean {
  return values.length === length && values.every(isFiniteNumber);
}

export function meshBounds(mesh: CollisionMesh): Aabb | null {
  const p = mesh.positions;
  if (p.length < 3) return null;
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i + 2 < p.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = p[i + k] as number;
      if (v < min[k]!) min[k] = v;
      if (v > max[k]!) max[k] = v;
    }
  }
  return { min, max };
}
