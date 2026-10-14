import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  CollisionAssetError,
  nodeTransformIssue,
  parseGlb,
  readCollisionGlb,
  writeCollisionGlb,
} from "../src/movement/glb";
import { layoutToCollisionMesh } from "../src/review/floorplan";
import type { CollisionMesh } from "../src/types";
import { sha256Hex } from "../src/manifest";
import { fixturePath, hasFixture, loadLayout, loadManifest } from "./helpers";

const quad: CollisionMesh = {
  positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]),
  indices: new Uint32Array([0, 2, 1, 0, 3, 2]),
};

function issuesOf(fn: () => unknown): string[] {
  try {
    fn();
  } catch (error) {
    if (error instanceof CollisionAssetError) return error.issues;
    throw error;
  }
  return [];
}

/** Rewrites the JSON chunk of a GLB (keeping the BIN chunk). */
function patchJson(glb: Uint8Array, patch: (json: Record<string, unknown>) => void): Uint8Array {
  const { json, bin } = parseGlb(glb);
  const obj = json as unknown as Record<string, unknown>;
  patch(obj);
  let jsonBytes = new TextEncoder().encode(JSON.stringify(obj));
  const pad = (4 - (jsonBytes.length % 4)) % 4;
  const padded = new Uint8Array(jsonBytes.length + pad).fill(0x20);
  padded.set(jsonBytes);
  jsonBytes = padded;
  const binBytes = bin ?? new Uint8Array();
  const total = 12 + 8 + jsonBytes.length + 8 + binBytes.length;
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, total, true);
  view.setUint32(12, jsonBytes.length, true);
  view.setUint32(16, 0x4e4f534a, true);
  out.set(jsonBytes, 20);
  view.setUint32(20 + jsonBytes.length, binBytes.length, true);
  view.setUint32(24 + jsonBytes.length, 0x004e4942, true);
  out.set(binBytes, 28 + jsonBytes.length);
  return out;
}

describe("collision GLB reader", () => {
  it("round-trips a valid world-space mesh", () => {
    const result = readCollisionGlb(writeCollisionGlb(quad));
    expect(Array.from(result.mesh.positions)).toEqual(Array.from(quad.positions));
    expect(Array.from(result.mesh.indices)).toEqual(Array.from(quad.indices));
    expect(result.stats.triangleCount).toBe(2);
    expect(result.stats.bounds).toEqual({ min: [0, 0, 0], max: [1, 0, 1] });
  });

  it("accepts explicit identity transforms", () => {
    const glb = writeCollisionGlb(quad, {
      matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    });
    expect(() => readCollisionGlb(glb)).not.toThrow();
    expect(() =>
      readCollisionGlb(
        writeCollisionGlb(quad, {
          translation: [0, 0, 0],
          rotation: [0, 0, 0, 1],
          scale: [1, 1, 1],
        }),
      ),
    ).not.toThrow();
  });

  it("rejects non-identity node transforms", () => {
    expect(
      issuesOf(() => readCollisionGlb(writeCollisionGlb(quad, { translation: [0, 1, 0] }))),
    ).toEqual([expect.stringContaining("translation is not zero")]);
    expect(
      issuesOf(() =>
        readCollisionGlb(writeCollisionGlb(quad, { rotation: [0, 0.7071068, 0, 0.7071068] })),
      ),
    ).toEqual([expect.stringContaining("rotation is not identity")]);
    expect(issuesOf(() => readCollisionGlb(writeCollisionGlb(quad, { scale: [2, 2, 2] })))).toEqual(
      [expect.stringContaining("scale is not one")],
    );
    expect(
      issuesOf(() =>
        readCollisionGlb(
          writeCollisionGlb(quad, { matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1] }),
        ),
      ),
    ).toEqual([expect.stringContaining("matrix is not identity")]);
  });

  it("rejects a non-identity transform on any node, even one without a mesh", () => {
    const glb = patchJson(writeCollisionGlb(quad), (json) => {
      (json.nodes as unknown[]).push({ name: "parent", translation: [1, 0, 0], children: [0] });
      json.scenes = [{ nodes: [1] }];
    });
    expect(issuesOf(() => readCollisionGlb(glb))[0]).toContain("node 1 (parent)");
  });

  it("rejects NaN vertices", () => {
    const bad: CollisionMesh = {
      positions: new Float32Array(quad.positions),
      indices: quad.indices,
    };
    bad.positions[4] = Number.NaN;
    expect(issuesOf(() => readCollisionGlb(writeCollisionGlb(bad)))[0]).toContain("non-finite");
    bad.positions[4] = Number.POSITIVE_INFINITY;
    expect(issuesOf(() => readCollisionGlb(writeCollisionGlb(bad)))[0]).toContain("non-finite");
  });

  it("rejects out-of-range indices", () => {
    const bad: CollisionMesh = { positions: quad.positions, indices: new Uint32Array([0, 1, 9]) };
    expect(issuesOf(() => readCollisionGlb(writeCollisionGlb(bad)))[0]).toContain("out of range");
  });

  it("rejects geometry outside the approved bounds", () => {
    const bounds = { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] };
    expect(issuesOf(() => readCollisionGlb(writeCollisionGlb(quad), { bounds }))[0]).toContain(
      "bounds",
    );
  });

  it("rejects malformed containers, strips and external buffers", () => {
    expect(() => parseGlb(new Uint8Array(40))).toThrow(CollisionAssetError);
    const strip = patchJson(writeCollisionGlb(quad), (json) => {
      (json.meshes as { primitives: { mode?: number }[] }[])[0]!.primitives[0]!.mode = 5;
    });
    expect(issuesOf(() => readCollisionGlb(strip))[0]).toContain("TRIANGLES");
    const external = patchJson(writeCollisionGlb(quad), (json) => {
      (json.buffers as { uri?: string }[])[0]!.uri = "https://example.com/evil.bin";
    });
    expect(issuesOf(() => readCollisionGlb(external))[0]).toContain("embedded GLB BIN");
  });

  it("checks node transforms independently", () => {
    expect(nodeTransformIssue({})).toBeNull();
    expect(nodeTransformIssue({ rotation: [0, 0, 0, -1] })).toBeNull();
    expect(nodeTransformIssue({ skin: 0 })).toContain("skinned");
  });

  it("round-trips the layout-generated collision mesh", () => {
    const mesh = layoutToCollisionMesh(loadLayout());
    const result = readCollisionGlb(writeCollisionGlb(mesh));
    expect(result.stats.triangleCount).toBe(mesh.indices.length / 3);
  });

  it.skipIf(!hasFixture("synthetic-two-room.collision.glb"))(
    "accepts the packaged fixture collision GLB within the manifest bounds",
    async () => {
      const bytes = readFileSync(fixturePath("synthetic-two-room.collision.glb"));
      const manifest = hasFixture("synthetic-two-room.manifest.json") ? loadManifest() : null;
      const result = readCollisionGlb(bytes, { bounds: manifest?.bounds ?? null });
      if (manifest) {
        expect(await sha256Hex(bytes)).toBe(manifest.assets.collision.sha256);
      }
      expect(result.stats.triangleCount).toBeGreaterThan(0);
      const b = result.stats.bounds!;
      expect(b.min[0]).toBeGreaterThanOrEqual(-4.2);
      expect(b.max[0]).toBeLessThanOrEqual(4.2);
    },
  );
});
