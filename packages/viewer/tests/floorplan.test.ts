import type { Layout } from "@andarilho/scene-contract";
import { describe, expect, it } from "vitest";
import {
  LayoutValidationError,
  groupProjectedArea,
  isDegeneratePolygon,
  isSelfIntersecting,
  layoutBounds,
  layoutToCollisionMesh,
  splitWall,
  validateLayout,
  type FloorplanMesh,
} from "../src/review/floorplan";
import type { Vec2 } from "../src/types";
import { loadLayout } from "./helpers";

function triangles(mesh: FloorplanMesh, kind: string, id?: string): number[][][] {
  const out: number[][][] = [];
  for (const g of mesh.groups) {
    if (g.kind !== kind || (id !== undefined && g.id !== id)) continue;
    for (let t = g.start; t < g.start + g.count; t += 3) {
      const tri: number[][] = [];
      for (let k = 0; k < 3; k++) {
        const v = mesh.indices[t + k]! * 3;
        tri.push([mesh.positions[v]!, mesh.positions[v + 1]!, mesh.positions[v + 2]!]);
      }
      out.push(tri);
    }
  }
  return out;
}

describe("fixture layout", () => {
  const layout = loadLayout();
  const mesh = layoutToCollisionMesh(layout);

  it("validates cleanly", () => {
    expect(validateLayout(layout).filter((i) => i.severity === "error")).toEqual([]);
  });

  it("covers the full 8 x 5 m floor facing up", () => {
    expect(groupProjectedArea(mesh, "floor")).toBeCloseTo(40, 6);
    for (const tri of triangles(mesh, "floor")) {
      const [a, b, c] = tri as [number[], number[], number[]];
      const ny = (b[2]! - a[2]!) * (c[0]! - a[0]!) - (b[0]! - a[0]!) * (c[2]! - a[2]!);
      expect(ny).toBeGreaterThan(0);
      expect(a[1]).toBe(0);
    }
  });

  it("splits the divider around the doorway with a lintel", () => {
    const divider = layout.walls!.find((w) => w.id === "wall-divider")!;
    const pieces = splitWall(divider, layout.openings!);
    expect(pieces).toEqual([
      { kind: "wall", id: "wall-divider", u0: 0, u1: 2.05, y0: 0, y1: 2.6 },
      { kind: "lintel", id: "door-0", u0: 2.05, u1: expect.closeTo(2.95, 9), y0: 2.1, y1: 2.6 },
      { kind: "wall", id: "wall-divider", u0: expect.closeTo(2.95, 9), u1: 5, y0: 0, y1: 2.6 },
    ]);
  });

  it("leaves the doorway gap open: no wall triangles across the opening below the lintel", () => {
    const doorZ: [number, number] = [-0.45, 0.45];
    for (const tri of triangles(mesh, "wall", "wall-divider")) {
      // Any wall triangle overlapping the door's z-range must lie outside it.
      const zs = tri.map((v) => v[2]!);
      const minZ = Math.min(...zs);
      const maxZ = Math.max(...zs);
      const overlaps = minZ < doorZ[1] - 1e-6 && maxZ > doorZ[0] + 1e-6;
      expect(overlaps).toBe(false);
    }
    const lintel = triangles(mesh, "lintel", "door-0");
    expect(lintel.length).toBe(12);
    for (const tri of lintel) {
      for (const v of tri) {
        expect(v[1]).toBeGreaterThanOrEqual(2.1 - 1e-6);
        expect(v[2]).toBeGreaterThanOrEqual(-0.45 - 1e-6);
        expect(v[2]).toBeLessThanOrEqual(0.45 + 1e-6);
      }
    }
  });

  it("extrudes the table obstacle to 0.75 m", () => {
    const tris = triangles(mesh, "obstacle", "table-0");
    expect(tris.length).toBe(12);
    const ys = tris.flat().map((v) => v[1]!);
    expect(Math.max(...ys)).toBeCloseTo(0.75);
    expect(Math.min(...ys)).toBeCloseTo(0);
    expect(groupProjectedArea(mesh, "obstacle")).toBeCloseTo(2, 6); // top + bottom caps
  });

  it("computes bounds including wall thickness and height", () => {
    const b = layoutBounds(layout);
    expect(b.min[0]).toBeCloseTo(-4.05);
    expect(b.max[1]).toBeCloseTo(2.6);
  });

  it("keeps a closed door solid", () => {
    const closed: Layout = {
      ...layout,
      openings: layout.openings!.map((o) => ({ ...o, state: "closed" as const })),
    };
    const pieces = splitWall(
      closed.walls!.find((w) => w.id === "wall-divider")!,
      closed.openings!,
    );
    expect(pieces).toEqual([{ kind: "wall", id: "wall-divider", u0: 0, u1: 5, y0: 0, y1: 2.6 }]);
  });

  it("adds a sill under a raised opening", () => {
    const wall = layout.walls![0]!;
    const pieces = splitWall(wall, [
      { id: "window", wallId: wall.id, offset: 1, width: 1, bottom: 0.9, top: 2 },
    ]);
    expect(pieces.map((p) => p.kind)).toEqual(["wall", "sill", "lintel", "wall"]);
  });
});

describe("floor holes", () => {
  it("triangulates a polygon with a hole with the correct area", () => {
    const layout: Layout = {
      schemaVersion: "1.0",
      floors: [
        {
          id: "f",
          elevation: 0.5,
          polygons: [
            [
              [0, 0],
              [6, 0],
              [6, 4],
              [0, 4],
            ],
          ],
          holes: [
            [
              [2, 1],
              [2, 3],
              [4, 3],
              [4, 1],
            ],
          ],
        },
      ],
      spawn: { feet: [1, 0.5, 1], yawRadians: 0, floorId: "f" },
    };
    const mesh = layoutToCollisionMesh(layout);
    expect(groupProjectedArea(mesh, "floor")).toBeCloseTo(24 - 4, 6);
    // No floor triangle centroid lies inside the hole.
    for (const tri of triangles(mesh, "floor")) {
      const cx = (tri[0]![0]! + tri[1]![0]! + tri[2]![0]!) / 3;
      const cz = (tri[0]![2]! + tri[1]![2]! + tri[2]![2]!) / 3;
      expect(cx > 2 && cx < 4 && cz > 1 && cz < 3).toBe(false);
      expect(tri[0]![1]).toBe(0.5);
    }
  });
});

describe("layout validation", () => {
  const bowtie: Vec2[] = [
    [0, 0],
    [2, 2],
    [2, 0],
    [0, 2],
  ];

  it("detects self-intersection and degenerate polygons", () => {
    expect(isSelfIntersecting(bowtie)).toBe(true);
    expect(
      isSelfIntersecting([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
      ]),
    ).toBe(false);
    expect(
      isDegeneratePolygon([
        [0, 0],
        [1, 0],
        [2, 0],
      ]),
    ).toBe(true);
    expect(
      isDegeneratePolygon([
        [0, 0],
        [1, 0],
      ]),
    ).toBe(true);
  });

  it("rejects bad layouts", () => {
    const base = loadLayout();
    const bad: Layout = {
      ...base,
      floors: [{ ...base.floors[0]!, polygons: [bowtie] }],
      walls: base.walls!.map((w, i) =>
        i === 0 ? { ...w, thickness: -0.1 } : i === 1 ? { ...w, height: 50 } : w,
      ),
      openings: [{ ...base.openings![0]!, offset: 4.5 }],
      obstacles: [{ ...base.obstacles![0]!, height: 0 }],
    };
    const messages = validateLayout(bad).map((i) => `${i.path}: ${i.message}`);
    expect(messages).toEqual(
      expect.arrayContaining([
        expect.stringContaining("self-intersects"),
        expect.stringContaining("thickness must be positive"),
        expect.stringContaining("impossible wall height"),
        expect.stringContaining("does not fit"),
        expect.stringContaining("impossible obstacle height"),
      ]),
    );
    expect(() => layoutToCollisionMesh(bad)).toThrow(LayoutValidationError);
  });

  it("rejects a spawn inside an obstacle or off the floor", () => {
    const base = loadLayout();
    const inTable = validateLayout({ ...base, spawn: { ...base.spawn, feet: [-2, 0, -0.5] } });
    expect(inTable.some((i) => i.message.includes("inside an obstacle"))).toBe(true);
    const offFloor = validateLayout({ ...base, spawn: { ...base.spawn, feet: [10, 0, 0] } });
    expect(offFloor.some((i) => i.message.includes("not on the walkable floor"))).toBe(true);
  });

  it("rejects overlapping openings and dangling wall references", () => {
    const base = loadLayout();
    const issues = validateLayout({
      ...base,
      openings: [
        ...base.openings!,
        { id: "door-1", wallId: "wall-divider", offset: 2.5, width: 0.9, top: 2 },
        { id: "door-2", wallId: "missing", offset: 0, width: 1, top: 2 },
      ],
    });
    expect(issues.some((i) => i.message.includes("overlaps"))).toBe(true);
    expect(issues.some((i) => i.message.includes("unknown wall"))).toBe(true);
  });
});
