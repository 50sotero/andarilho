// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LayoutCanvas } from "@/features/review/LayoutCanvas";
import { LayoutPlan } from "@/features/review/LayoutPlan";
import type { Furniture, Layout } from "@/lib/api/types";
import fixture from "@andarilho/scene-contract/fixtures/synthetic-two-room.layout.json";

const base = fixture as unknown as Layout;
const viewBox = { minX: -5, minY: -5, width: 10, height: 10 };

function piece(overrides: Partial<Furniture> = {}): Furniture {
  return {
    id: "chair-0",
    floorId: "floor-0",
    category: "chair",
    label: "Wooden chair",
    // Clear of the fixture's spawn and table.
    position: [1, 1],
    yawRadians: 0,
    size: [1, 1, 2],
    solid: true,
    parts: [{ shape: "box", min: [0, 0, 0], max: [1, 1, 1], color: "#aa3333" }],
    ...overrides,
  };
}

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => {
    root.unmount();
  });
  container.remove();
});

function renderCanvas(
  layout: Layout,
  onSelect = vi.fn(),
  selection: Parameters<typeof LayoutCanvas>[0]["selection"] = null,
) {
  act(() => {
    root.render(
      <LayoutCanvas
        layout={layout}
        viewBox={viewBox}
        selection={selection}
        onSelect={onSelect}
        onChange={vi.fn()}
        showOverlay={false}
      />,
    );
  });
  return onSelect;
}

/** The footprint corners of a drawn piece, as numbers. */
function corners(id: string): [number, number][] {
  const polygon = container.querySelector(`[data-furniture="${id}"]`);
  if (!polygon) throw new Error(`no polygon for ${id}`);
  return (polygon.getAttribute("points") ?? "")
    .split(" ")
    .map((pair) => pair.split(",").map(Number) as [number, number]);
}

function expectCorners(actual: [number, number][], expected: [number, number][]): void {
  expect(actual).toHaveLength(expected.length);
  expected.forEach(([x, z], i) => {
    expect(actual[i]![0]).toBeCloseTo(x, 6);
    expect(actual[i]![1]).toBeCloseTo(z, 6);
  });
}

describe("furniture footprints in the editor", () => {
  it("draws a piece at yaw 0 with its width along x and its depth along z", () => {
    renderCanvas({ ...base, furniture: [piece()] });
    // Centre (1, 1), width 1 (x 0.5 to 1.5), depth 2 (z 0 to 2).
    expectCorners(corners("chair-0"), [
      [0.5, 0],
      [1.5, 0],
      [1.5, 2],
      [0.5, 2],
    ]);
  });

  it("draws a piece at yaw pi/2 with its front facing +x and its depth along x", () => {
    renderCanvas({ ...base, furniture: [piece({ yawRadians: Math.PI / 2 })] });
    // The depth now runs along x (0 to 2) and the width along z (0.5 to 1.5).
    expectCorners(corners("chair-0"), [
      [0, 1.5],
      [0, 0.5],
      [2, 0.5],
      [2, 1.5],
    ]);
  });

  it("gives each piece its label as a tooltip, or its category without a label", () => {
    renderCanvas({
      ...base,
      furniture: [
        piece(),
        piece({ id: "shelf-0", label: undefined, category: "shelf", position: [-2.5, 2] }),
      ],
    });
    expect(container.querySelector('[data-furniture="chair-0"] title')?.textContent).toBe(
      "Wooden chair",
    );
    expect(container.querySelector('[data-furniture="shelf-0"] title')?.textContent).toBe("shelf");
  });

  it("dashes the outline of a piece the user can walk through, and not a solid one", () => {
    renderCanvas({
      ...base,
      furniture: [piece(), piece({ id: "rug-0", solid: false, position: [-2.5, 2] })],
    });
    const solid = container.querySelector('[data-furniture="chair-0"]')!;
    const passable = container.querySelector('[data-furniture="rug-0"]')!;
    expect(solid.getAttribute("stroke-dasharray")).toBeNull();
    expect(passable.getAttribute("stroke-dasharray")).toBeTruthy();
    expect(passable.getAttribute("fill-opacity")).toBe("0.4");
    expect(passable.getAttribute("aria-label")).toContain("does not block walking");
  });

  it("selects a piece with a pointer press or with Enter", () => {
    const onSelect = renderCanvas({ ...base, furniture: [piece()] });
    const polygon = container.querySelector('[data-furniture="chair-0"]')!;
    act(() => {
      polygon.dispatchEvent(new Event("pointerdown", { bubbles: true }));
    });
    expect(onSelect).toHaveBeenLastCalledWith({ kind: "furniture", index: 0 });
    act(() => {
      polygon.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    });
    expect(onSelect).toHaveBeenLastCalledWith({ kind: "furniture", index: 0 });
  });

  it("marks the selected piece as pressed", () => {
    renderCanvas({ ...base, furniture: [piece()] }, vi.fn(), { kind: "furniture", index: 0 });
    expect(
      container.querySelector('[data-furniture="chair-0"]')?.getAttribute("aria-pressed"),
    ).toBe("true");
  });
});

describe("furniture on the read-only plan", () => {
  it("draws pieces with the same footprints and dashes the passable ones", () => {
    act(() => {
      root.render(
        <LayoutPlan
          layout={{
            ...base,
            furniture: [piece(), piece({ id: "rug-0", solid: false, position: [-2.5, 2] })],
          }}
          title="Flat"
        />,
      );
    });
    expectCorners(corners("chair-0"), [
      [0.5, 0],
      [1.5, 0],
      [1.5, 2],
      [0.5, 2],
    ]);
    expect(
      container.querySelector('[data-furniture="rug-0"]')?.getAttribute("stroke-dasharray"),
    ).toBeTruthy();
    expect(
      container.querySelector('[data-furniture="chair-0"]')?.getAttribute("stroke-dasharray"),
    ).toBeNull();
  });
});
