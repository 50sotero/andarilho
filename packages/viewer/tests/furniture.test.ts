import type { Furniture, Layout } from "@andarilho/scene-contract";
import { Vector3 } from "three";
import { describe, expect, it } from "vitest";
import {
  LayoutValidationError,
  MAX_FURNITURE_HEIGHT,
  MAX_FURNITURE_WIDTH,
  furnitureFootprint,
  furniturePartCorner,
  groupProjectedArea,
  layoutBounds,
  layoutToCollisionMesh,
  validateLayout,
  type FloorplanMesh,
} from "../src/review/floorplan";
import { loadLayout } from "./helpers";

const RED = "#aa3333";

/** A one-room layout: a 6 x 4 m floor at `elevation`, with the spawn at (0.5, 0.5). */
function room(elevation = 0): Layout {
  return {
    schemaVersion: "1.0",
    floors: [
      {
        id: "floor-0",
        elevation,
        polygons: [
          [
            [0, 0],
            [6, 0],
            [6, 4],
            [0, 4],
          ],
        ],
      },
    ],
    spawn: { feet: [0.5, elevation, 0.5], yawRadians: 0, floorId: "floor-0" },
  };
}

/** A 0.5 x 0.9 x 0.5 m chair with one box part, centred at (3, 2) and facing +z. */
function chair(overrides: Partial<Furniture> = {}): Furniture {
  return {
    id: "chair-0",
    floorId: "floor-0",
    category: "chair",
    label: "wooden chair",
    position: [3, 2],
    yawRadians: 0,
    size: [0.5, 0.9, 0.5],
    parts: [{ shape: "box", min: [0, 0, 0], max: [1, 1, 1], color: RED }],
    ...overrides,
  };
}

function withFurniture(...items: Furniture[]): Layout {
  return { ...room(), furniture: items };
}

function errorsOf(layout: Layout): string[] {
  return validateLayout(layout)
    .filter((i) => i.severity === "error")
    .map((i) => `${i.path}: ${i.message}`);
}

/** Lowest and highest vertex heights of one collision group. */
function groupHeights(mesh: FloorplanMesh, kind: string, id: string): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const g of mesh.groups) {
    if (g.kind !== kind || g.id !== id) continue;
    for (let t = g.start; t < g.start + g.count; t++) {
      const y = mesh.positions[mesh.indices[t]! * 3 + 1]!;
      lo = Math.min(lo, y);
      hi = Math.max(hi, y);
    }
  }
  return [lo, hi];
}

/** Distinct (x, z) vertex positions of one collision group, rounded to 1 mm. */
function groupPlanPoints(mesh: FloorplanMesh, kind: string, id: string): string[] {
  const seen = new Set<string>();
  for (const g of mesh.groups) {
    if (g.kind !== kind || g.id !== id) continue;
    for (let t = g.start; t < g.start + g.count; t++) {
      const v = mesh.indices[t]! * 3;
      const x = mesh.positions[v]!.toFixed(3);
      const z = mesh.positions[v + 2]!.toFixed(3);
      seen.add(`${x},${z}`);
    }
  }
  return [...seen].sort();
}

function expectClose(actual: readonly number[], expected: readonly number[]): void {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((v, i) => expect(actual[i]).toBeCloseTo(v, 9));
}

describe("furniture frame", () => {
  it("keeps the footprint unrotated at yaw 0", () => {
    const corners = furnitureFootprint(chair());
    expectClose(corners[0]!, [2.75, 1.75]);
    expectClose(corners[1]!, [3.25, 1.75]);
    expectClose(corners[2]!, [3.25, 2.25]);
    expectClose(corners[3]!, [2.75, 2.25]);
  });

  it("turns the front to +x at yaw pi/2", () => {
    const item = chair({ position: [0, 0], size: [1, 1, 2], yawRadians: Math.PI / 2 });
    // The front (w = 1) faces +x and the back (w = 0) faces -x; the object's +x
    // (u = 1) points to -z.
    const front = furniturePartCorner(item, [0.5, 0, 1], 0);
    expectClose([front[0], front[2]], [1, 0]);
    const back = furniturePartCorner(item, [0.5, 0, 0], 0);
    expectClose([back[0], back[2]], [-1, 0]);
    const right = furniturePartCorner(item, [1, 0, 0.5], 0);
    expectClose([right[0], right[2]], [0, -0.5]);
  });

  it("matches three.js rotation.y for an arbitrary yaw", () => {
    const yaw = 0.7;
    const item = chair({ position: [1.5, -2], size: [0.8, 1, 0.6], yawRadians: yaw });
    const [u, v, w] = [0.2, 0.5, 0.9];
    const world = furniturePartCorner(item, [u, v, w], 0);
    const local = new Vector3((u - 0.5) * 0.8, 0.5, (w - 0.5) * 0.6);
    const turned = local.applyAxisAngle(new Vector3(0, 1, 0), yaw);
    expectClose(world, [1.5 + turned.x, 0.5, -2 + turned.z]);
  });

  it("places a part corner at the floor elevation plus its height", () => {
    const top = furniturePartCorner(chair(), [1, 1, 1], 0.5);
    expectClose(top, [3.25, 1.4, 2.25]);
  });

  it("keeps every footprint corner the same distance from the centre at any yaw", () => {
    for (const yaw of [0, 0.4, 2.1]) {
      const corners = furnitureFootprint(chair({ yawRadians: yaw }));
      for (const [x, z] of corners) {
        expect(Math.hypot(x! - 3, z! - 2)).toBeCloseTo(Math.hypot(0.25, 0.25), 9);
      }
    }
  });
});

describe("furniture in the collision mesh", () => {
  it("adds one prism per solid piece, from its floor up to its height", () => {
    const mesh = layoutToCollisionMesh(withFurniture(chair()));
    const groups = mesh.groups.filter((g) => g.kind === "furniture");
    expect(groups.map((g) => g.id)).toEqual(["chair-0"]);
    const [lo, hi] = groupHeights(mesh, "furniture", "chair-0");
    // The mesh stores float32 vertices, so mesh values are compared to 5 digits.
    expect(lo).toBeCloseTo(0, 5);
    expect(hi).toBeCloseTo(0.9, 5);
    // Top and bottom caps, each the footprint area.
    expect(groupProjectedArea(mesh, "furniture")).toBeCloseTo(2 * 0.5 * 0.5, 5);
  });

  it("stands the prism on the floor it belongs to", () => {
    const mesh = layoutToCollisionMesh({ ...room(0.5), furniture: [chair()] });
    const [lo, hi] = groupHeights(mesh, "furniture", "chair-0");
    expect(lo).toBeCloseTo(0.5, 5);
    expect(hi).toBeCloseTo(1.4, 5);
  });

  it("builds the prism from the rotated footprint", () => {
    const item = chair({ yawRadians: Math.PI / 4 });
    const mesh = layoutToCollisionMesh(withFurniture(item));
    const expected = furnitureFootprint(item).map(([x, z]) => `${x.toFixed(3)},${z.toFixed(3)}`);
    expect(groupPlanPoints(mesh, "furniture", "chair-0")).toEqual([...expected].sort());
    expect(groupProjectedArea(mesh, "furniture")).toBeCloseTo(2 * 0.5 * 0.5, 5);
  });

  it("adds no collision for a piece that is not solid", () => {
    const mesh = layoutToCollisionMesh(withFurniture(chair({ solid: false })));
    expect(mesh.groups.some((g) => g.kind === "furniture")).toBe(false);
  });

  it("treats a piece without a solid flag as solid", () => {
    const item = chair();
    delete item.solid;
    const mesh = layoutToCollisionMesh(withFurniture(item));
    expect(mesh.groups.some((g) => g.kind === "furniture" && g.id === "chair-0")).toBe(true);
  });

  it("refuses to build collision from an invalid piece", () => {
    expect(() => layoutToCollisionMesh(withFurniture(chair({ size: [0, 1, 1] })))).toThrow(
      LayoutValidationError,
    );
  });
});

describe("layout bounds with furniture", () => {
  it("extends the bounds to a piece outside the room, including its height", () => {
    const bounds = layoutBounds(withFurniture(chair({ position: [7, 3], size: [1, 2, 1] })));
    expect(bounds.max[0]).toBeCloseTo(7.5, 9);
    expect(bounds.max[1]).toBeCloseTo(2, 9);
  });

  it("keeps a piece that is not solid in the bounds", () => {
    const bounds = layoutBounds(
      withFurniture(chair({ position: [7, 3], size: [1, 2, 1], solid: false })),
    );
    expect(bounds.max[0]).toBeCloseTo(7.5, 9);
  });
});

describe("furniture validation", () => {
  it("accepts a well-formed piece on the fixture layout", () => {
    const layout: Layout = { ...loadLayout(), furniture: [chair({ position: [-1, 1.5] })] };
    expect(errorsOf(layout)).toEqual([]);
  });

  it("rejects a piece on an unknown floor", () => {
    expect(errorsOf(withFurniture(chair({ floorId: "floor-9" })))).toEqual([
      "furniture[0].floorId: unknown floor floor-9",
    ]);
  });

  it("rejects an id shared with an opening, an obstacle or another piece", () => {
    const base = loadLayout();
    expect(errorsOf({ ...base, furniture: [chair({ id: "door-0" })] })).toEqual([
      "furniture[0].id: duplicate id door-0",
    ]);
    expect(errorsOf({ ...base, furniture: [chair({ id: "table-0" })] })).toEqual([
      "furniture[0].id: duplicate id table-0",
    ]);
    expect(errorsOf(withFurniture(chair(), chair()))).toEqual([
      "furniture[1].id: duplicate id chair-0",
    ]);
  });

  it("rejects a size that is not three positive numbers", () => {
    for (const size of [
      [0.5, 0, 0.5],
      [0, 1, 1],
      [-1, 1, 1],
      [Number.NaN, 1, 1],
    ] as [number, number, number][]) {
      expect(errorsOf(withFurniture(chair({ size })))).toEqual([
        "furniture[0].size: size must be three positive numbers",
      ]);
    }
  });

  it("rejects furniture beyond the plausible limits, and accepts the limits themselves", () => {
    expect(errorsOf(withFurniture(chair({ size: [MAX_FURNITURE_WIDTH + 0.5, 1, 1] })))).toEqual([
      "furniture[0].size: furniture is implausibly large",
    ]);
    expect(errorsOf(withFurniture(chair({ size: [1, 1, MAX_FURNITURE_WIDTH + 0.5] })))).toEqual([
      "furniture[0].size: furniture is implausibly large",
    ]);
    expect(errorsOf(withFurniture(chair({ size: [1, MAX_FURNITURE_HEIGHT + 0.5, 1] })))).toEqual([
      "furniture[0].size: furniture is implausibly large",
    ]);
    expect(
      errorsOf(
        withFurniture(
          chair({
            position: [5, 2],
            size: [MAX_FURNITURE_WIDTH, MAX_FURNITURE_HEIGHT, MAX_FURNITURE_WIDTH],
          }),
        ),
      ),
    ).toEqual([]);
  });

  it("rejects a part whose min is not below its max on every axis", () => {
    const flat = chair({
      parts: [{ shape: "box", min: [0, 0, 0], max: [0, 1, 1], color: RED }],
    });
    expect(errorsOf(withFurniture(flat))).toEqual([
      "furniture[0].parts[0]: part min must be below max on every axis",
    ]);
  });

  it("rejects part corners outside the object box", () => {
    const outside = chair({
      parts: [{ shape: "box", min: [0, 0, 0], max: [1.2, 1, 1], color: RED }],
    });
    expect(errorsOf(withFurniture(outside))).toEqual([
      "furniture[0].parts[0]: part corners must be numbers from 0 to 1",
    ]);
    const negative = chair({
      parts: [{ shape: "box", min: [-0.1, 0, 0], max: [1, 1, 1], color: RED }],
    });
    expect(errorsOf(withFurniture(negative))).toEqual([
      "furniture[0].parts[0]: part corners must be numbers from 0 to 1",
    ]);
  });

  it("rejects a colour or shape that is not allowed", () => {
    const badColour = chair({
      parts: [{ shape: "box", min: [0, 0, 0], max: [1, 1, 1], color: "red" }],
    });
    expect(errorsOf(withFurniture(badColour))).toEqual([
      "furniture[0].parts[0].color: color must be #rrggbb",
    ]);
    const badShape = chair({
      parts: [{ shape: "sphere" as "box", min: [0, 0, 0], max: [1, 1, 1], color: RED }],
    });
    expect(errorsOf(withFurniture(badShape))).toEqual([
      "furniture[0].parts[0].shape: shape must be a box or a cylinder",
    ]);
  });

  it("requires between one and 32 parts", () => {
    const unitPart = (): Furniture["parts"][number] => ({
      shape: "box",
      min: [0, 0, 0],
      max: [1, 1, 1],
      color: RED,
    });
    const none = chair({ parts: [] });
    expect(errorsOf(withFurniture(none))).toEqual([
      "furniture[0].parts: furniture needs between 1 and 32 parts",
    ]);
    const many = chair({ parts: Array.from({ length: 33 }, unitPart) });
    expect(errorsOf(withFurniture(many))).toEqual([
      "furniture[0].parts: furniture needs between 1 and 32 parts",
    ]);
    const thirtyTwo = chair({ parts: Array.from({ length: 32 }, unitPart) });
    expect(errorsOf(withFurniture(thirtyTwo))).toEqual([]);
  });

  it("rejects a non-finite position or yaw without failing the rest of the check", () => {
    expect(errorsOf(withFurniture(chair({ position: [Number.NaN, 1] })))).toEqual(
      expect.arrayContaining(["furniture[0].position: position is not finite"]),
    );
    expect(errorsOf(withFurniture(chair({ yawRadians: Number.POSITIVE_INFINITY })))).toEqual([
      "furniture[0].yawRadians: yaw is not finite",
    ]);
  });

  it("rejects an overlong category or label", () => {
    expect(errorsOf(withFurniture(chair({ category: "x".repeat(41) })))).toEqual([
      "furniture[0].category: category must be 1 to 40 characters",
    ]);
    expect(errorsOf(withFurniture(chair({ label: "y".repeat(121) })))).toEqual([
      "furniture[0].label: label must be at most 120 characters",
    ]);
  });

  it("rejects a spawn inside solid furniture", () => {
    const blocking = chair({ position: [0.5, 0.5] });
    expect(errorsOf(withFurniture(blocking))).toEqual([
      "spawn: spawn is inside a piece of furniture",
    ]);
  });

  it("lets the spawn stand in furniture that is not solid", () => {
    const passable = chair({ position: [0.5, 0.5], solid: false });
    expect(errorsOf(withFurniture(passable))).toEqual([]);
  });
});
