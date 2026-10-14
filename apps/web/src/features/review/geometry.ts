/**
 * Small pure 2D helpers for the layout editor. Coordinates are canonical world
 * meters on the horizontal plane: a layout point [x, z] maps to world (x, y, z).
 */
import type { Layout } from "@/lib/api/types";

export type Point = [number, number];

export function distance(a: Point, b: Point): number {
  return Math.hypot(b[0] - a[0], b[1] - a[1]);
}

export function polygonSignedArea(ring: readonly Point[]): number {
  let sum = 0;
  for (let i = 0; i < ring.length; i += 1) {
    const a = ring[i]!;
    const b = ring[(i + 1) % ring.length]!;
    sum += a[0] * b[1] - b[0] * a[1];
  }
  return sum / 2;
}

export function polygonArea(ring: readonly Point[]): number {
  return Math.abs(polygonSignedArea(ring));
}

function orientation(a: Point, b: Point, c: Point): number {
  const v = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1]);
  if (Math.abs(v) < 1e-12) return 0;
  return v > 0 ? 1 : 2;
}

function onSegment(a: Point, b: Point, c: Point): boolean {
  return (
    Math.min(a[0], c[0]) - 1e-12 <= b[0] &&
    b[0] <= Math.max(a[0], c[0]) + 1e-12 &&
    Math.min(a[1], c[1]) - 1e-12 <= b[1] &&
    b[1] <= Math.max(a[1], c[1]) + 1e-12
  );
}

export function segmentsIntersect(p1: Point, q1: Point, p2: Point, q2: Point): boolean {
  const o1 = orientation(p1, q1, p2);
  const o2 = orientation(p1, q1, q2);
  const o3 = orientation(p2, q2, p1);
  const o4 = orientation(p2, q2, q1);
  if (o1 !== o2 && o3 !== o4) return true;
  if (o1 === 0 && onSegment(p1, p2, q1)) return true;
  if (o2 === 0 && onSegment(p1, q2, q1)) return true;
  if (o3 === 0 && onSegment(p2, p1, q2)) return true;
  if (o4 === 0 && onSegment(p2, q1, q2)) return true;
  return false;
}

/** True when any two non-adjacent edges of the ring intersect. */
export function isSelfIntersecting(ring: readonly Point[]): boolean {
  const n = ring.length;
  for (let i = 0; i < n; i += 1) {
    const a1 = ring[i]!;
    const a2 = ring[(i + 1) % n]!;
    for (let j = i + 1; j < n; j += 1) {
      if (Math.abs(i - j) <= 1 || (i === 0 && j === n - 1)) continue;
      const b1 = ring[j]!;
      const b2 = ring[(j + 1) % n]!;
      if (segmentsIntersect(a1, a2, b1, b2)) return true;
    }
  }
  return false;
}

export function pointInPolygon(p: Point, ring: readonly Point[]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i, i += 1) {
    const a = ring[i]!;
    const b = ring[j]!;
    if (
      a[1] > p[1] !== b[1] > p[1] &&
      p[0] < ((b[0] - a[0]) * (p[1] - a[1])) / (b[1] - a[1]) + a[0]
    ) {
      inside = !inside;
    }
  }
  return inside;
}

export interface LayoutProblem {
  message: string;
}

/** Client-side sanity checks before saving; the API validates again. */
export function layoutProblems(layout: Layout): LayoutProblem[] {
  const problems: LayoutProblem[] = [];
  const floorIds = new Set(layout.floors.map((f) => f.id));
  for (const floor of layout.floors) {
    floor.polygons.forEach((ring, i) => {
      if (ring.length < 3 || polygonArea(ring as Point[]) < 0.01) {
        problems.push({ message: `Floor ${floor.id} polygon ${i + 1} is degenerate.` });
      } else if (isSelfIntersecting(ring as Point[])) {
        problems.push({ message: `Floor ${floor.id} polygon ${i + 1} crosses itself.` });
      }
    });
  }
  for (const wall of layout.walls ?? []) {
    if (!(wall.thickness > 0))
      problems.push({ message: `Wall ${wall.id} needs a positive thickness.` });
    if (!(wall.height > 0) || wall.height > 20) {
      problems.push({ message: `Wall ${wall.id} height must be between 0 and 20 m.` });
    }
    if (distance(wall.start, wall.end) < 0.01)
      problems.push({ message: `Wall ${wall.id} has no length.` });
  }
  const walls = new Map((layout.walls ?? []).map((w) => [w.id, w]));
  for (const opening of layout.openings ?? []) {
    const wall = walls.get(opening.wallId);
    if (!wall) {
      problems.push({ message: `Opening ${opening.id} refers to a missing wall.` });
      continue;
    }
    const len = distance(wall.start, wall.end);
    if (!(opening.width > 0) || opening.offset < 0 || opening.offset + opening.width > len + 1e-6) {
      problems.push({ message: `Opening ${opening.id} does not fit inside wall ${wall.id}.` });
    }
    if (!(opening.top > (opening.bottom ?? 0))) {
      problems.push({ message: `Opening ${opening.id} top must be above its bottom.` });
    }
  }
  for (const obstacle of layout.obstacles ?? []) {
    if (obstacle.footprint.length < 3 || polygonArea(obstacle.footprint as Point[]) < 1e-4) {
      problems.push({ message: `Obstacle ${obstacle.id} footprint is degenerate.` });
    }
    if (!(obstacle.height > 0))
      problems.push({ message: `Obstacle ${obstacle.id} needs a positive height.` });
  }
  if (!floorIds.has(layout.spawn.floorId)) {
    problems.push({ message: "The spawn point must be on an existing floor." });
  } else {
    const floor = layout.floors.find((f) => f.id === layout.spawn.floorId)!;
    const p: Point = [layout.spawn.feet[0], layout.spawn.feet[2]];
    if (!floor.polygons.some((ring) => pointInPolygon(p, ring as Point[]))) {
      problems.push({ message: "The spawn point must be inside the floor outline." });
    }
  }
  return problems;
}

/** Uniformly scales all horizontal coordinates and heights about the origin. */
export function scaleLayout(layout: Layout, factor: number): Layout {
  if (!(factor > 0) || !Number.isFinite(factor)) return layout;
  const s = (p: [number, number]): [number, number] => [p[0] * factor, p[1] * factor];
  return {
    ...layout,
    floors: layout.floors.map((f) => ({
      ...f,
      elevation: f.elevation * factor,
      polygons: f.polygons.map((r) => r.map(s)),
      holes: f.holes?.map((r) => r.map(s)),
    })),
    walls: layout.walls?.map((w) => ({
      ...w,
      start: s(w.start),
      end: s(w.end),
      thickness: w.thickness * factor,
    })),
    openings: layout.openings?.map((o) => ({
      ...o,
      offset: o.offset * factor,
      width: o.width * factor,
    })),
    obstacles: layout.obstacles?.map((o) => ({ ...o, footprint: o.footprint.map(s) })),
    landmarks: layout.landmarks?.map((l) => ({
      ...l,
      worldPoint: [l.worldPoint[0] * factor, l.worldPoint[1] * factor, l.worldPoint[2] * factor],
    })),
    spawn: {
      ...layout.spawn,
      feet: [
        layout.spawn.feet[0] * factor,
        layout.spawn.feet[1] * factor,
        layout.spawn.feet[2] * factor,
      ],
    },
  };
}

/** Scale factor so that a measured length becomes the known real length. */
export function calibrationFactor(measured: number, known: number): number | null {
  if (!(measured > 0) || !(known > 0) || !Number.isFinite(measured) || !Number.isFinite(known))
    return null;
  return known / measured;
}

export interface ViewBox {
  minX: number;
  minY: number;
  width: number;
  height: number;
}

export function layoutViewBox(layout: Layout, padding = 0.5): ViewBox {
  const pts: Point[] = [];
  for (const f of layout.floors) for (const r of f.polygons) pts.push(...(r as Point[]));
  for (const w of layout.walls ?? []) pts.push(w.start, w.end);
  for (const o of layout.obstacles ?? []) pts.push(...(o.footprint as Point[]));
  pts.push([layout.spawn.feet[0], layout.spawn.feet[2]]);
  if (pts.length === 0) return { minX: -5, minY: -5, width: 10, height: 10 };
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const minX = Math.min(...xs) - padding;
  const minY = Math.min(...ys) - padding;
  return {
    minX,
    minY,
    width: Math.max(1, Math.max(...xs) + padding - minX),
    height: Math.max(1, Math.max(...ys) + padding - minY),
  };
}

/** Point along a wall at `offset` meters from its start. */
export function pointAlong(start: Point, end: Point, offset: number): Point {
  const len = distance(start, end) || 1;
  const t = offset / len;
  return [start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t];
}

export function round(value: number, digits = 3): number {
  const f = 10 ** digits;
  return Math.round(value * f) / f;
}
