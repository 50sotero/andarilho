import type { Layout, LayoutFloor, Opening, Wall } from "@andarilho/scene-contract";
import { ShapeUtils, Vector2 } from "three";
import type { Aabb, CollisionMesh, Vec2, Vec3 } from "../types";

/**
 * Confirmed layout -> world-space collision triangle soup. Mirrors the worker's
 * collision packaging: floor polygons with holes are triangulated (ear
 * clipping), walls are extruded with thickness/height and split around open
 * openings with a lintel (and sill) instead of a solid wall across the
 * opening, and obstacles are extruded prisms. Layout 2D points are (x, z) in
 * the canonical world frame; Y is up. Nothing here applies the render
 * transform: layout coordinates are already world coordinates.
 */

export type FloorplanPartKind = "floor" | "wall" | "lintel" | "sill" | "obstacle";

export interface FloorplanGroup {
  kind: FloorplanPartKind;
  /** Source element ID (floor, wall or obstacle ID; opening ID for lintel/sill). */
  id: string;
  /** First index (into `indices`) of this group. */
  start: number;
  /** Number of indices in this group (3 per triangle). */
  count: number;
}

export interface FloorplanMesh extends CollisionMesh {
  groups: FloorplanGroup[];
}

export interface LayoutIssue {
  severity: "error" | "warning";
  path: string;
  message: string;
}

export class LayoutValidationError extends Error {
  readonly issues: LayoutIssue[];
  constructor(issues: LayoutIssue[]) {
    super(
      `Invalid layout: ${issues
        .map((i) => `${i.path}: ${i.message}`)
        .slice(0, 5)
        .join("; ")}`,
    );
    this.name = "LayoutValidationError";
    this.issues = issues;
  }
}

/** Maximum plausible wall/obstacle height in meters for a single-floor release. */
export const MAX_ELEMENT_HEIGHT = 10;
const EPS = 1e-9;

// ---------------------------------------------------------------------------
// 2D polygon helpers
// ---------------------------------------------------------------------------

/** Removes a duplicated closing vertex and consecutive duplicates. */
export function normalizeRing(ring: readonly Vec2[]): Vec2[] {
  const out: Vec2[] = [];
  for (const p of ring) {
    const last = out[out.length - 1];
    if (!last || Math.abs(last[0] - p[0]) > EPS || Math.abs(last[1] - p[1]) > EPS) {
      out.push([p[0], p[1]]);
    }
  }
  if (out.length > 1) {
    const first = out[0]!;
    const last = out[out.length - 1]!;
    if (Math.abs(first[0] - last[0]) <= EPS && Math.abs(first[1] - last[1]) <= EPS) out.pop();
  }
  return out;
}

/** Shoelace signed area in the (x, z) plane. */
export function polygonSignedArea(ring: readonly Vec2[]): number {
  let sum = 0;
  for (let i = 0; i < ring.length; i++) {
    const a = ring[i]!;
    const b = ring[(i + 1) % ring.length]!;
    sum += a[0] * b[1] - b[0] * a[1];
  }
  return sum / 2;
}

export function polygonArea(ring: readonly Vec2[]): number {
  return Math.abs(polygonSignedArea(ring));
}

function orient(a: Vec2, b: Vec2, c: Vec2): number {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

function onSegment(a: Vec2, b: Vec2, p: Vec2): boolean {
  return (
    Math.min(a[0], b[0]) - EPS <= p[0] &&
    p[0] <= Math.max(a[0], b[0]) + EPS &&
    Math.min(a[1], b[1]) - EPS <= p[1] &&
    p[1] <= Math.max(a[1], b[1]) + EPS
  );
}

/** True when closed segments ab and cd intersect (including touching/collinear overlap). */
export function segmentsIntersect(a: Vec2, b: Vec2, c: Vec2, d: Vec2): boolean {
  const o1 = orient(a, b, c);
  const o2 = orient(a, b, d);
  const o3 = orient(c, d, a);
  const o4 = orient(c, d, b);
  const s = (v: number) => (Math.abs(v) <= EPS ? 0 : Math.sign(v));
  if (s(o1) * s(o2) < 0 && s(o3) * s(o4) < 0) return true;
  if (s(o1) === 0 && onSegment(a, b, c)) return true;
  if (s(o2) === 0 && onSegment(a, b, d)) return true;
  if (s(o3) === 0 && onSegment(c, d, a)) return true;
  if (s(o4) === 0 && onSegment(c, d, b)) return true;
  return false;
}

/** True when any two non-adjacent edges of the ring intersect. */
export function isSelfIntersecting(input: readonly Vec2[]): boolean {
  const ring = normalizeRing(input);
  const n = ring.length;
  if (n < 4) return false;
  for (let i = 0; i < n; i++) {
    const a = ring[i]!;
    const b = ring[(i + 1) % n]!;
    for (let j = i + 1; j < n; j++) {
      if (j === i || (j + 1) % n === i || j === (i + 1) % n) continue;
      const c = ring[j]!;
      const d = ring[(j + 1) % n]!;
      if (segmentsIntersect(a, b, c, d)) return true;
    }
  }
  return false;
}

/** True when the ring has fewer than 3 distinct vertices or (near) zero area. */
export function isDegeneratePolygon(input: readonly Vec2[], minArea = 1e-6): boolean {
  const ring = normalizeRing(input);
  return ring.length < 3 || polygonArea(ring) < minArea;
}

/** Even-odd point-in-polygon test. */
export function pointInPolygon(p: Vec2, ring: readonly Vec2[]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const a = ring[i]!;
    const b = ring[j]!;
    if (a[1] > p[1] !== b[1] > p[1]) {
      const x = ((b[0] - a[0]) * (p[1] - a[1])) / (b[1] - a[1]) + a[0];
      if (p[0] < x) inside = !inside;
    }
  }
  return inside;
}

function finitePoint(p: unknown): p is Vec2 {
  return (
    Array.isArray(p) &&
    p.length === 2 &&
    typeof p[0] === "number" &&
    typeof p[1] === "number" &&
    Number.isFinite(p[0]) &&
    Number.isFinite(p[1])
  );
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

function ringIssues(ring: readonly Vec2[], path: string, issues: LayoutIssue[]): boolean {
  if (!Array.isArray(ring) || !ring.every(finitePoint)) {
    issues.push({ severity: "error", path, message: "polygon has non-finite points" });
    return false;
  }
  if (normalizeRing(ring).length < 3) {
    issues.push({ severity: "error", path, message: "polygon is degenerate" });
    return false;
  }
  if (isSelfIntersecting(ring)) {
    issues.push({ severity: "error", path, message: "polygon self-intersects" });
    return false;
  }
  if (isDegeneratePolygon(ring)) {
    issues.push({ severity: "error", path, message: "polygon is degenerate" });
    return false;
  }
  return true;
}

function wallLength(wall: Wall): number {
  return Math.hypot(wall.end[0] - wall.start[0], wall.end[1] - wall.start[1]);
}

/**
 * Validates a confirmed layout: degenerate and self-intersecting polygons,
 * dangling references, negative thickness, impossible heights, openings that do
 * not fit their wall or overlap, and a spawn outside the walkable floor.
 */
export function validateLayout(layout: Layout): LayoutIssue[] {
  const issues: LayoutIssue[] = [];
  const err = (path: string, message: string) => issues.push({ severity: "error", path, message });
  if (layout.schemaVersion !== "1.0") err("schemaVersion", "unsupported layout schema version");
  if (!Array.isArray(layout.floors) || layout.floors.length === 0) err("floors", "no floors");

  const floors = new Map<string, LayoutFloor>();
  (layout.floors ?? []).forEach((floor, f) => {
    const path = `floors[${f}]`;
    if (floors.has(floor.id)) err(`${path}.id`, `duplicate floor id ${floor.id}`);
    floors.set(floor.id, floor);
    if (!Number.isFinite(floor.elevation)) err(`${path}.elevation`, "elevation is not finite");
    if (!floor.polygons?.length) err(`${path}.polygons`, "floor has no polygons");
    const valid: Vec2[][] = [];
    (floor.polygons ?? []).forEach((poly, p) => {
      if (ringIssues(poly, `${path}.polygons[${p}]`, issues)) valid.push(normalizeRing(poly));
    });
    (floor.holes ?? []).forEach((hole, h) => {
      const hp = `${path}.holes[${h}]`;
      if (!ringIssues(hole, hp, issues)) return;
      const ring = normalizeRing(hole);
      const owner = valid.find((poly) => ring.every((pt) => pointInPolygon(pt, poly)));
      if (!owner) err(hp, "hole is not inside any floor polygon");
    });
  });

  const walls = new Map<string, Wall>();
  (layout.walls ?? []).forEach((wall, w) => {
    const path = `walls[${w}]`;
    if (walls.has(wall.id)) err(`${path}.id`, `duplicate wall id ${wall.id}`);
    walls.set(wall.id, wall);
    if (!floors.has(wall.floorId)) err(`${path}.floorId`, `unknown floor ${wall.floorId}`);
    if (!finitePoint(wall.start) || !finitePoint(wall.end)) {
      err(path, "wall endpoints are not finite");
      return;
    }
    if (!(wallLength(wall) > 1e-3)) err(path, "wall has zero length");
    if (!(wall.thickness > 0)) err(`${path}.thickness`, "thickness must be positive");
    if (!(wall.height > 0) || wall.height > MAX_ELEMENT_HEIGHT) {
      err(`${path}.height`, "impossible wall height");
    }
  });

  const byWall = new Map<string, { opening: Opening; index: number }[]>();
  (layout.openings ?? []).forEach((opening, o) => {
    const path = `openings[${o}]`;
    const wall = walls.get(opening.wallId);
    if (!wall) {
      err(`${path}.wallId`, `unknown wall ${opening.wallId}`);
      return;
    }
    const bottom = opening.bottom ?? 0;
    const values = [opening.offset, opening.width, bottom, opening.top];
    if (!values.every((v) => Number.isFinite(v))) {
      err(path, "opening values are not finite");
      return;
    }
    if (!(opening.width > 0)) err(`${path}.width`, "width must be positive");
    if (opening.offset < 0 || opening.offset + opening.width > wallLength(wall) + 1e-6) {
      err(path, "opening does not fit within its wall");
    }
    if (bottom < 0 || !(opening.top > bottom) || opening.top > wall.height + 1e-6) {
      err(path, "impossible opening heights");
    }
    const list = byWall.get(wall.id) ?? [];
    list.push({ opening, index: o });
    byWall.set(wall.id, list);
  });
  for (const list of byWall.values()) {
    const sorted = [...list].sort((a, b) => a.opening.offset - b.opening.offset);
    for (let i = 1; i < sorted.length; i++) {
      const prev = sorted[i - 1]!.opening;
      if (sorted[i]!.opening.offset < prev.offset + prev.width - 1e-6) {
        err(`openings[${sorted[i]!.index}]`, "opening overlaps another opening on the same wall");
      }
    }
  }

  (layout.obstacles ?? []).forEach((obstacle, o) => {
    const path = `obstacles[${o}]`;
    if (!floors.has(obstacle.floorId)) err(`${path}.floorId`, `unknown floor ${obstacle.floorId}`);
    ringIssues(obstacle.footprint, `${path}.footprint`, issues);
    if (!(obstacle.height > 0) || obstacle.height > MAX_ELEMENT_HEIGHT) {
      err(`${path}.height`, "impossible obstacle height");
    }
  });

  const spawn = layout.spawn;
  if (!spawn || !Array.isArray(spawn.feet) || !spawn.feet.every((v) => Number.isFinite(v))) {
    err("spawn.feet", "spawn feet are not finite");
  } else if (!Number.isFinite(spawn.yawRadians)) {
    err("spawn.yawRadians", "spawn yaw is not finite");
  } else {
    const floor = floors.get(spawn.floorId);
    if (!floor) {
      err("spawn.floorId", `unknown floor ${spawn.floorId}`);
    } else {
      const p: Vec2 = [spawn.feet[0], spawn.feet[2]];
      const onFloor = (floor.polygons ?? []).some((poly) => pointInPolygon(p, normalizeRing(poly)));
      const inHole = (floor.holes ?? []).some((h) => pointInPolygon(p, normalizeRing(h)));
      if (!onFloor || inHole) err("spawn", "spawn is not on the walkable floor");
      const inObstacle = (layout.obstacles ?? []).some(
        (ob) => ob.floorId === floor.id && pointInPolygon(p, normalizeRing(ob.footprint)),
      );
      if (inObstacle) err("spawn", "spawn is inside an obstacle");
      if (Math.abs(spawn.feet[1] - floor.elevation) > 0.5) {
        issues.push({
          severity: "warning",
          path: "spawn.feet",
          message: "spawn height differs from the floor elevation",
        });
      }
    }
  }
  return issues;
}

// ---------------------------------------------------------------------------
// Mesh generation
// ---------------------------------------------------------------------------

class MeshBuilder {
  readonly positions: number[] = [];
  readonly indices: number[] = [];
  readonly groups: FloorplanGroup[] = [];
  private current: FloorplanGroup | null = null;

  begin(kind: FloorplanPartKind, id: string): void {
    this.end();
    this.current = { kind, id, start: this.indices.length, count: 0 };
  }

  end(): void {
    if (this.current) {
      this.current.count = this.indices.length - this.current.start;
      if (this.current.count > 0) this.groups.push(this.current);
      this.current = null;
    }
  }

  private vertex(p: Vec3): number {
    this.positions.push(p[0], p[1], p[2]);
    return this.positions.length / 3 - 1;
  }

  /** Adds a triangle wound so its normal points along `desired`. */
  triangle(a: Vec3, b: Vec3, c: Vec3, desired: Vec3): void {
    const ux = b[0] - a[0];
    const uy = b[1] - a[1];
    const uz = b[2] - a[2];
    const vx = c[0] - a[0];
    const vy = c[1] - a[1];
    const vz = c[2] - a[2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    const dot = nx * desired[0] + ny * desired[1] + nz * desired[2];
    const ia = this.vertex(a);
    const ib = this.vertex(b);
    const ic = this.vertex(c);
    if (dot >= 0) this.indices.push(ia, ib, ic);
    else this.indices.push(ia, ic, ib);
  }

  quad(a: Vec3, b: Vec3, c: Vec3, d: Vec3, desired: Vec3): void {
    this.triangle(a, b, c, desired);
    this.triangle(a, c, d, desired);
  }

  /** Horizontal cap at height y (normal +Y or -Y) triangulated with holes. */
  cap(outer: Vec2[], holes: Vec2[][], y: number, up: boolean): void {
    const contour = outer.map((p) => new Vector2(p[0], p[1]));
    const holeVecs = holes.map((h) => h.map((p) => new Vector2(p[0], p[1])));
    const all = [...outer, ...holes.flat()];
    const faces = ShapeUtils.triangulateShape(contour, holeVecs);
    const desired: Vec3 = [0, up ? 1 : -1, 0];
    for (const face of faces) {
      const [i, j, k] = face as [number, number, number];
      const a = all[i]!;
      const b = all[j]!;
      const c = all[k]!;
      this.triangle([a[0], y, a[1]], [b[0], y, b[1]], [c[0], y, c[1]], desired);
    }
  }

  /** Vertical prism over a simple footprint between y0 and y1, outward normals. */
  prism(footprint: Vec2[], y0: number, y1: number): void {
    const ring = normalizeRing(footprint);
    if (ring.length < 3 || !(y1 > y0)) return;
    const ccw = polygonSignedArea(ring) > 0;
    for (let i = 0; i < ring.length; i++) {
      const p = ring[i]!;
      const q = ring[(i + 1) % ring.length]!;
      const ex = q[0] - p[0];
      const ez = q[1] - p[1];
      // Outward normal in (x, z): right of the edge for CCW, left for CW.
      const outward: Vec3 = ccw ? [ez, 0, -ex] : [-ez, 0, ex];
      this.quad([p[0], y0, p[1]], [q[0], y0, q[1]], [q[0], y1, q[1]], [p[0], y1, p[1]], outward);
    }
    this.cap(ring, [], y1, true);
    this.cap(ring, [], y0, false);
  }

  build(): FloorplanMesh {
    this.end();
    return {
      positions: new Float32Array(this.positions),
      indices: new Uint32Array(this.indices),
      groups: this.groups,
    };
  }
}

/** Footprint quad of a wall interval [u0, u1] along the wall centerline. */
export function wallPieceFootprint(wall: Wall, u0: number, u1: number): Vec2[] {
  const len = wallLength(wall);
  const dx = (wall.end[0] - wall.start[0]) / len;
  const dz = (wall.end[1] - wall.start[1]) / len;
  const h = wall.thickness / 2;
  const nx = -dz * h;
  const nz = dx * h;
  const at = (u: number, side: number): Vec2 => [
    wall.start[0] + dx * u + nx * side,
    wall.start[1] + dz * u + nz * side,
  ];
  return [at(u0, -1), at(u1, -1), at(u1, 1), at(u0, 1)];
}

export interface WallPiece {
  kind: "wall" | "lintel" | "sill";
  id: string;
  /** Interval along the wall centerline from the start point, meters. */
  u0: number;
  u1: number;
  /** Heights above the floor elevation, meters. */
  y0: number;
  y1: number;
}

/**
 * Splits a wall into solid pieces around its open openings: full-height
 * segments on either side, a lintel above (when the opening top is below the
 * wall top) and a sill below (when the opening bottom is above the floor).
 * Closed openings stay solid: a visibly closed door is a blocked boundary.
 */
export function splitWall(wall: Wall, openings: readonly Opening[]): WallPiece[] {
  const len = wallLength(wall);
  const open = openings
    .filter((o) => o.wallId === wall.id && (o.state ?? "open") === "open")
    .sort((a, b) => a.offset - b.offset);
  const pieces: WallPiece[] = [];
  let cursor = 0;
  for (const o of open) {
    const u0 = Math.max(cursor, Math.max(0, o.offset));
    const u1 = Math.min(len, o.offset + o.width);
    if (u0 > cursor + 1e-6) {
      pieces.push({ kind: "wall", id: wall.id, u0: cursor, u1: u0, y0: 0, y1: wall.height });
    }
    if (u1 > u0) {
      const bottom = Math.max(0, o.bottom ?? 0);
      const top = Math.min(wall.height, o.top);
      if (bottom > 1e-6) pieces.push({ kind: "sill", id: o.id, u0, u1, y0: 0, y1: bottom });
      if (top < wall.height - 1e-6) {
        pieces.push({ kind: "lintel", id: o.id, u0, u1, y0: top, y1: wall.height });
      }
    }
    cursor = Math.max(cursor, u1);
  }
  if (len > cursor + 1e-6) {
    pieces.push({ kind: "wall", id: wall.id, u0: cursor, u1: len, y0: 0, y1: wall.height });
  }
  return pieces;
}

export interface FloorplanOptions {
  /** Validate first and throw LayoutValidationError on errors (default true). */
  validate?: boolean;
}

/** Builds the world-space collision triangle soup for a confirmed layout. */
export function layoutToCollisionMesh(
  layout: Layout,
  options: FloorplanOptions = {},
): FloorplanMesh {
  if (options.validate ?? true) {
    const errors = validateLayout(layout).filter((i) => i.severity === "error");
    if (errors.length > 0) throw new LayoutValidationError(errors);
  }
  const builder = new MeshBuilder();
  const elevation = new Map(layout.floors.map((f) => [f.id, f.elevation]));

  for (const floor of layout.floors) {
    const polygons = floor.polygons.map((p) => normalizeRing(p));
    const holes = (floor.holes ?? []).map((h) => normalizeRing(h));
    polygons.forEach((poly) => {
      const own = holes.filter((h) => h.every((pt) => pointInPolygon(pt, poly)));
      builder.begin("floor", floor.id);
      builder.cap(poly, own, floor.elevation, true);
    });
  }

  for (const wall of layout.walls ?? []) {
    const base = elevation.get(wall.floorId) ?? 0;
    for (const piece of splitWall(wall, layout.openings ?? [])) {
      builder.begin(piece.kind, piece.id);
      builder.prism(wallPieceFootprint(wall, piece.u0, piece.u1), base + piece.y0, base + piece.y1);
    }
  }

  for (const obstacle of layout.obstacles ?? []) {
    const base = elevation.get(obstacle.floorId) ?? 0;
    builder.begin("obstacle", obstacle.id);
    builder.prism(obstacle.footprint, base, base + obstacle.height);
  }

  return builder.build();
}

/** Axis-aligned bounds of a layout (floors, walls and obstacles), in world meters. */
export function layoutBounds(layout: Layout): Aabb {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  const add = (x: number, y: number, z: number) => {
    min[0] = Math.min(min[0], x);
    min[1] = Math.min(min[1], y);
    min[2] = Math.min(min[2], z);
    max[0] = Math.max(max[0], x);
    max[1] = Math.max(max[1], y);
    max[2] = Math.max(max[2], z);
  };
  const elevation = new Map(layout.floors.map((f) => [f.id, f.elevation]));
  for (const floor of layout.floors) {
    for (const poly of floor.polygons) for (const p of poly) add(p[0], floor.elevation, p[1]);
  }
  for (const wall of layout.walls ?? []) {
    const base = elevation.get(wall.floorId) ?? 0;
    for (const p of wallPieceFootprint(wall, 0, wallLength(wall))) {
      add(p[0], base, p[1]);
      add(p[0], base + wall.height, p[1]);
    }
  }
  for (const ob of layout.obstacles ?? []) {
    const base = elevation.get(ob.floorId) ?? 0;
    for (const p of ob.footprint) add(p[0], base + ob.height, p[1]);
  }
  return { min, max };
}

/** Sums the area of the triangles of a group kind projected onto the XZ plane. */
export function groupProjectedArea(mesh: FloorplanMesh, kind: FloorplanPartKind): number {
  let area = 0;
  for (const group of mesh.groups) {
    if (group.kind !== kind) continue;
    for (let t = group.start; t < group.start + group.count; t += 3) {
      const a = mesh.indices[t]! * 3;
      const b = mesh.indices[t + 1]! * 3;
      const c = mesh.indices[t + 2]! * 3;
      const p = mesh.positions;
      area += Math.abs(
        ((p[b]! - p[a]!) * (p[c + 2]! - p[a + 2]!) - (p[c]! - p[a]!) * (p[b + 2]! - p[a + 2]!)) / 2,
      );
    }
  }
  return area;
}
