import { describe, expect, it } from "vitest";
import { deliveryEnvelopeIssues, resolveViewerSource, sceneManifestIssues } from "../src/manifest";
import { decomposeRenderTransform, RenderTransformError } from "../src/render/transform";
import { canWalk, transition, type ViewerState } from "../src/viewer-state";
import { hasFixture, loadManifest, syntheticManifest } from "./helpers";

describe("worldFromDecodedRender", () => {
  it("decomposes a column-major rotation + translation + uniform scale", () => {
    const c = Math.cos(0.5);
    const s = Math.sin(0.5);
    const k = 2;
    // Rotation about +Y by 0.5 rad, scale 2, translation (1, 2, 3), column-major.
    const m = [c * k, 0, -s * k, 0, 0, k, 0, 0, s * k, 0, c * k, 0, 1, 2, 3, 1];
    const t = decomposeRenderTransform(m);
    expect(t.scale).toBeCloseTo(2);
    expect(t.position.toArray()).toEqual([1, 2, 3]);
    expect(t.quaternion.y).toBeCloseTo(Math.sin(0.25));
  });

  it("rejects reflections, non-uniform scale, shear and projective rows", () => {
    const id = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const reflect = [...id];
    reflect[0] = -1;
    expect(() => decomposeRenderTransform(reflect)).toThrow(/reflection/);
    const stretch = [...id];
    stretch[5] = 2;
    expect(() => decomposeRenderTransform(stretch)).toThrow(/uniform/);
    const shear = [...id];
    shear[4] = 0.5;
    expect(() => decomposeRenderTransform(shear)).toThrow(RenderTransformError);
    const projective = [...id];
    projective[3] = 0.1;
    expect(() => decomposeRenderTransform(projective)).toThrow(/affine/);
    expect(() => decomposeRenderTransform([1, 2, 3])).toThrow(/16/);
  });
});

describe("manifest and envelope checks", () => {
  it("accepts the synthetic manifest", () => {
    expect(sceneManifestIssues(syntheticManifest())).toEqual([]);
  });

  it("rejects wrong frames and spawn outside bounds", () => {
    const bad = syntheticManifest({
      geometrySpace: "local" as "world",
      spawn: { feet: [100, 0, 0], yawRadians: 0, floorId: "floor-0" },
    });
    const issues = sceneManifestIssues(bad);
    expect(issues).toEqual(
      expect.arrayContaining(["geometrySpace must be world", "spawn is outside the scene bounds"]),
    );
  });

  it("rejects mismatched revisions and missing render URLs", () => {
    const manifest = syntheticManifest();
    const issues = deliveryEnvelopeIssues({
      sceneRevisionId: "other",
      manifestSha256: "",
      expiresAt: new Date(Date.now() + 1000).toISOString(),
      manifest,
      assetUrls: {},
    });
    expect(issues).toEqual(
      expect.arrayContaining([
        "envelope and manifest refer to different scene revisions",
        "render asset URL is missing or invalid",
      ]),
    );
  });

  it("resolves dev URL sources", () => {
    const r = resolveViewerSource({
      kind: "urls",
      manifest: syntheticManifest(),
      renderUrl: "/fixtures/scene.ply",
      collisionUrl: "/fixtures/collision.glb",
    });
    expect(r.renderFormat).toBe("ply");
    expect(r.collisionUrl).toBe("/fixtures/collision.glb");
    expect(() =>
      resolveViewerSource({
        kind: "urls",
        manifest: syntheticManifest(),
        renderUrl: "javascript:alert(1)",
      }),
    ).toThrow();
  });

  it.skipIf(!hasFixture("synthetic-two-room.manifest.json"))("accepts the fixture manifest", () => {
    const manifest = loadManifest();
    expect(sceneManifestIssues(manifest)).toEqual([]);
    expect(manifest.worldFromDecodedRender).toEqual([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    ]);
  });
});

describe("viewer state machine", () => {
  const ready: ViewerState = {
    status: "ready",
    mode: "walkable",
    inspectionReason: null,
    quality: "desktop",
  };

  it("follows idle -> loading -> ready -> walking", () => {
    let s: ViewerState = { status: "idle" };
    s = transition(s, { type: "load-start" });
    expect(s.status).toBe("loading");
    s = transition(s, { type: "progress", progress: { render: "done" } });
    expect(s).toMatchObject({ progress: { render: "done", collision: "pending" } });
    s = transition(s, {
      type: "loaded",
      mode: "walkable",
      inspectionReason: null,
      quality: "mobile",
    });
    expect(canWalk(s)).toBe(true);
    s = transition(s, { type: "enter-walk", pointerLocked: false });
    expect(s.status).toBe("walking");
    s = transition(s, { type: "exit-walk" });
    expect(s).toMatchObject({ status: "ready", mode: "walkable" });
  });

  it("never walks from inspection, loading or error", () => {
    const inspection: ViewerState = {
      ...ready,
      mode: "inspection",
      inspectionReason: "no collision",
    };
    expect(transition(inspection, { type: "enter-walk", pointerLocked: false })).toBe(inspection);
    const loading = transition({ status: "idle" }, { type: "load-start" });
    expect(transition(loading, { type: "enter-walk", pointerLocked: false })).toBe(loading);
    expect(canWalk(inspection)).toBe(false);
  });

  it("handles context loss and disposal as terminal", () => {
    const walking = transition(ready, { type: "enter-walk", pointerLocked: true });
    const lost = transition(walking, { type: "context-lost" });
    expect(lost).toMatchObject({ status: "context-lost", resumeMode: "walkable" });
    expect(transition(lost, { type: "load-start" }).status).toBe("loading");
    const disposed = transition(lost, { type: "dispose" });
    expect(transition(disposed, { type: "load-start" })).toBe(disposed);
  });
});
