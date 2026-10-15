import { describe, expect, it } from "vitest";
import { furnitureFootprint } from "@andarilho/viewer";
import { layoutProblems, layoutViewBox, scaleLayout } from "@/features/review/geometry";
import {
  addObstacle,
  describeSelection,
  removeSelected,
  type Selection,
} from "@/features/review/layout-edit";
import type { Furniture, Layout } from "@/lib/api/types";
import fixture from "@andarilho/scene-contract/fixtures/synthetic-two-room.layout.json";

const base = fixture as unknown as Layout;

function piece(overrides: Partial<Furniture> = {}): Furniture {
  return {
    id: "furniture-1",
    floorId: "floor-0",
    category: "chair",
    label: "Wooden chair",
    position: [-1, 1.5],
    yawRadians: 0,
    size: [0.5, 0.9, 0.5],
    solid: true,
    parts: [{ shape: "box", min: [0, 0, 0], max: [1, 1, 1], color: "#aa3333" }],
    ...overrides,
  };
}

function withPieces(...items: Furniture[]): Layout {
  return { ...base, furniture: items };
}

describe("furniture in the review checks", () => {
  it("flags a spawn inside a solid piece, as the viewer does", () => {
    const layout = withPieces(piece({ position: [2, 0] }));
    expect(layoutProblems(layout).map((p) => p.message)).toEqual([
      "The spawn point must not be inside a piece of furniture.",
    ]);
  });

  it("allows the spawn inside a piece that is not solid", () => {
    const layout = withPieces(piece({ position: [2, 0], solid: false }));
    expect(layoutProblems(layout)).toEqual([]);
  });

  it("accepts a layout whose furniture stands clear of the spawn", () => {
    expect(layoutProblems(withPieces(piece()))).toEqual([]);
  });
});

describe("furniture in scaling and framing", () => {
  it("scales the place and size of a piece, and leaves its parts as fractions", () => {
    const scaled = scaleLayout(withPieces(piece()), 2);
    const [item] = scaled.furniture!;
    expect(item!.position).toEqual([-2, 3]);
    expect(item!.size).toEqual([1, 1.8, 1]);
    expect(item!.parts).toEqual(piece().parts);
  });

  it("keeps a scaled piece on the same spot of the scaled room", () => {
    const before = furnitureFootprint(piece());
    const after = furnitureFootprint(scaleLayout(withPieces(piece()), 2).furniture![0]!);
    before.forEach(([x, z], i) => {
      expect(after[i]![0]).toBeCloseTo(x * 2, 9);
      expect(after[i]![1]).toBeCloseTo(z * 2, 9);
    });
  });

  it("frames a piece that stands outside the room", () => {
    const outside = withPieces(piece({ position: [6, 0], size: [2, 1, 2] }));
    const view = layoutViewBox(outside);
    expect(view.minX + view.width).toBeGreaterThanOrEqual(7);
  });
});

describe("removing furniture", () => {
  it("removes the selected piece and keeps the others", () => {
    const layout = withPieces(piece({ id: "a" }), piece({ id: "b" }));
    const sel: Selection = { kind: "furniture", index: 0 };
    const next = removeSelected(layout, sel);
    expect(next.furniture?.map((f) => f.id)).toEqual(["b"]);
    // The edit returns a new layout and leaves the one it was given untouched.
    expect(layout.furniture).toHaveLength(2);
  });

  it("does nothing for a selection that names no piece", () => {
    const layout = withPieces(piece());
    expect(removeSelected(layout, { kind: "furniture", index: 5 })).toBe(layout);
  });

  it("describes a piece by its label, or by its category when it has none", () => {
    const named = withPieces(piece());
    expect(describeSelection(named, { kind: "furniture", index: 0 })).toBe(
      "Furniture Wooden chair",
    );
    const unnamed = withPieces(piece({ label: undefined }));
    expect(describeSelection(unnamed, { kind: "furniture", index: 0 })).toBe("Furniture chair");
  });

  it("keeps new obstacle ids clear of furniture ids", () => {
    const layout = withPieces(piece({ id: "obstacle-1" }));
    const next = addObstacle(layout, [0, 0]);
    // The fixture's own obstacle stays; the new one skips the furniture's id.
    expect(next.obstacles?.map((o) => o.id)).toEqual(["table-0", "obstacle-2"]);
  });
});
