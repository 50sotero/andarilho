/** Pure editing operations on a Layout (immutable: every function returns a new layout). */
import type { Layout } from "@/lib/api/types";
import { distance, round, type Point } from "./geometry";

export type Selection =
  | { kind: "floorCorner"; floor: number; ring: number; index: number }
  | { kind: "wall"; index: number }
  | { kind: "wallEnd"; index: number; end: "start" | "end" }
  | { kind: "opening"; index: number }
  | { kind: "obstacle"; index: number }
  | { kind: "obstacleCorner"; index: number; corner: number }
  | { kind: "furniture"; index: number }
  | { kind: "spawn" }
  | { kind: "landmark"; index: number };

function clone(layout: Layout): Layout {
  return structuredClone(layout);
}

const r = (p: Point): [number, number] => [round(p[0]), round(p[1])];

/** Moves the point behind a draggable selection. Non-point selections are ignored. */
export function movePoint(layout: Layout, sel: Selection, p: Point): Layout {
  const next = clone(layout);
  switch (sel.kind) {
    case "floorCorner": {
      const ring = next.floors[sel.floor]?.polygons[sel.ring];
      if (ring && ring[sel.index]) ring[sel.index] = r(p);
      break;
    }
    case "wallEnd": {
      const wall = next.walls?.[sel.index];
      if (wall) wall[sel.end] = r(p);
      break;
    }
    case "obstacleCorner": {
      const ob = next.obstacles?.[sel.index];
      if (ob && ob.footprint[sel.corner]) ob.footprint[sel.corner] = r(p);
      break;
    }
    case "spawn":
      next.spawn.feet = [round(p[0]), next.spawn.feet[1], round(p[1])];
      break;
    case "landmark": {
      const lm = next.landmarks?.[sel.index];
      if (lm) lm.worldPoint = [round(p[0]), lm.worldPoint[1], round(p[1])];
      break;
    }
    default:
      return layout;
  }
  return next;
}

export function pointOf(layout: Layout, sel: Selection): Point | null {
  switch (sel.kind) {
    case "floorCorner":
      return (
        (layout.floors[sel.floor]?.polygons[sel.ring]?.[sel.index] as Point | undefined) ?? null
      );
    case "wallEnd":
      return (layout.walls?.[sel.index]?.[sel.end] as Point | undefined) ?? null;
    case "obstacleCorner":
      return (layout.obstacles?.[sel.index]?.footprint[sel.corner] as Point | undefined) ?? null;
    case "spawn":
      return [layout.spawn.feet[0], layout.spawn.feet[2]];
    case "landmark": {
      const lm = layout.landmarks?.[sel.index];
      return lm ? [lm.worldPoint[0], lm.worldPoint[2]] : null;
    }
    default:
      return null;
  }
}

export function uniqueId(prefix: string, existing: Iterable<string>): string {
  const used = new Set(existing);
  let n = 1;
  while (used.has(`${prefix}-${n}`)) n += 1;
  return `${prefix}-${n}`;
}

function allIds(layout: Layout): string[] {
  return [
    ...layout.floors.map((f) => f.id),
    ...(layout.walls ?? []).map((w) => w.id),
    ...(layout.openings ?? []).map((o) => o.id),
    ...(layout.obstacles ?? []).map((o) => o.id),
    ...(layout.furniture ?? []).map((f) => f.id),
    ...(layout.landmarks ?? []).map((l) => l.id),
  ];
}

export function addWall(layout: Layout, center: Point): Layout {
  const next = clone(layout);
  const floorId = layout.spawn.floorId || layout.floors[0]?.id || "floor-0";
  next.walls = [
    ...(next.walls ?? []),
    {
      id: uniqueId("wall", allIds(layout)),
      floorId,
      start: r([center[0] - 1, center[1]]),
      end: r([center[0] + 1, center[1]]),
      thickness: 0.1,
      height: 2.5,
      provenance: "user",
    },
  ];
  return next;
}

export function addOpening(layout: Layout, wallIndex: number): Layout {
  const wall = layout.walls?.[wallIndex];
  if (!wall) return layout;
  const len = distance(wall.start, wall.end);
  const width = Math.min(0.9, Math.max(0.1, len * 0.5));
  const next = clone(layout);
  next.openings = [
    ...(next.openings ?? []),
    {
      id: uniqueId("opening", allIds(layout)),
      wallId: wall.id,
      offset: round(Math.max(0, (len - width) / 2)),
      width: round(width),
      bottom: 0,
      top: 2.1,
      state: "open",
      provenance: "user",
    },
  ];
  return next;
}

export function addObstacle(layout: Layout, center: Point): Layout {
  const next = clone(layout);
  const floorId = layout.spawn.floorId || layout.floors[0]?.id || "floor-0";
  const [x, z] = center;
  next.obstacles = [
    ...(next.obstacles ?? []),
    {
      id: uniqueId("obstacle", allIds(layout)),
      floorId,
      footprint: [
        r([x - 0.5, z - 0.5]),
        r([x + 0.5, z - 0.5]),
        r([x + 0.5, z + 0.5]),
        r([x - 0.5, z + 0.5]),
      ],
      height: 0.8,
      provenance: "user",
    },
  ];
  return next;
}

export function addLandmark(layout: Layout, center: Point): Layout {
  const next = clone(layout);
  next.landmarks = [
    ...(next.landmarks ?? []),
    {
      id: uniqueId("landmark", allIds(layout)),
      planPoint: null,
      worldPoint: [round(center[0]), 0, round(center[1])],
      uncertainty: 0.1,
    },
  ];
  return next;
}

/** Inserts a corner at the midpoint of the edge after the given corner. */
export function insertFloorCorner(
  layout: Layout,
  floor: number,
  ring: number,
  index: number,
): Layout {
  const pts = layout.floors[floor]?.polygons[ring];
  if (!pts || pts.length < 2) return layout;
  const a = pts[index]!;
  const b = pts[(index + 1) % pts.length]!;
  const next = clone(layout);
  next.floors[floor]!.polygons[ring]!.splice(
    index + 1,
    0,
    r([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]),
  );
  return next;
}

/** Removes the selected element. Floor polygons keep at least 3 corners. */
export function removeSelected(layout: Layout, sel: Selection): Layout {
  const next = clone(layout);
  switch (sel.kind) {
    case "floorCorner": {
      const ring = next.floors[sel.floor]?.polygons[sel.ring];
      if (!ring || ring.length <= 3) return layout;
      ring.splice(sel.index, 1);
      return next;
    }
    case "wall":
    case "wallEnd": {
      const wall = next.walls?.[sel.index];
      if (!wall) return layout;
      next.walls = next.walls!.filter((_, i) => i !== sel.index);
      next.openings = (next.openings ?? []).filter((o) => o.wallId !== wall.id);
      return next;
    }
    case "opening":
      next.openings = (next.openings ?? []).filter((_, i) => i !== sel.index);
      return next;
    case "obstacle":
    case "obstacleCorner":
      next.obstacles = (next.obstacles ?? []).filter((_, i) => i !== sel.index);
      return next;
    case "furniture":
      if (!next.furniture?.[sel.index]) return layout;
      next.furniture = next.furniture.filter((_, i) => i !== sel.index);
      return next;
    case "landmark":
      next.landmarks = (next.landmarks ?? []).filter((_, i) => i !== sel.index);
      return next;
    default:
      return layout;
  }
}

/** Updates a numeric/enum field of the element behind a selection. */
export function updateField(
  layout: Layout,
  sel: Selection,
  field: string,
  value: number | string,
): Layout {
  const next = clone(layout);
  const num = typeof value === "number" ? value : Number(value);
  switch (sel.kind) {
    case "wall":
    case "wallEnd": {
      const w = next.walls?.[sel.index];
      if (!w) return layout;
      if (field === "thickness" || field === "height") w[field] = num;
      else if (field === "length") {
        // Keep the start fixed and move the end along the wall direction.
        const len = distance(w.start, w.end) || 1;
        const k = num / len;
        w.end = r([
          w.start[0] + (w.end[0] - w.start[0]) * k,
          w.start[1] + (w.end[1] - w.start[1]) * k,
        ]);
      }
      return next;
    }
    case "opening": {
      const o = next.openings?.[sel.index];
      if (!o) return layout;
      if (field === "state") o.state = value === "closed" ? "closed" : "open";
      else if (field === "offset" || field === "width" || field === "bottom" || field === "top")
        o[field] = num;
      return next;
    }
    case "obstacle":
    case "obstacleCorner": {
      const o = next.obstacles?.[sel.index];
      if (!o) return layout;
      if (field === "height") o.height = num;
      return next;
    }
    case "spawn":
      if (field === "yawDegrees") next.spawn.yawRadians = round((num * Math.PI) / 180, 4);
      return next;
    case "landmark": {
      const l = next.landmarks?.[sel.index];
      if (!l) return layout;
      if (field === "y") l.worldPoint = [l.worldPoint[0], num, l.worldPoint[2]];
      else if (field === "uncertainty") l.uncertainty = num;
      return next;
    }
    default:
      return layout;
  }
}

export function describeSelection(layout: Layout, sel: Selection): string {
  switch (sel.kind) {
    case "floorCorner":
      return `Floor ${layout.floors[sel.floor]?.id ?? ""} corner ${sel.index + 1}`;
    case "wall":
      return `Wall ${layout.walls?.[sel.index]?.id ?? ""}`;
    case "wallEnd":
      return `Wall ${layout.walls?.[sel.index]?.id ?? ""} ${sel.end} point`;
    case "opening":
      return `Opening ${layout.openings?.[sel.index]?.id ?? ""}`;
    case "obstacle":
      return `Obstacle ${layout.obstacles?.[sel.index]?.id ?? ""}`;
    case "obstacleCorner":
      return `Obstacle ${layout.obstacles?.[sel.index]?.id ?? ""} corner ${sel.corner + 1}`;
    case "furniture": {
      const item = layout.furniture?.[sel.index];
      return `Furniture ${item?.label || item?.category || ""}`.trimEnd();
    }
    case "spawn":
      return "Starting point";
    case "landmark":
      return `Landmark ${layout.landmarks?.[sel.index]?.id ?? ""}`;
  }
}
