import type { DeliveryEnvelope, SceneManifest } from "@andarilho/scene-contract";
import { describe, expect, it } from "vitest";
import {
  deliveryEnvelopeIssues,
  OUTDATED_SCENE_MESSAGE,
  OutdatedSceneError,
  resolveViewerSource,
  sceneManifestIssues,
} from "../src/manifest";
import { canWalk, transition, type ViewerState } from "../src/viewer-state";
import { hasFixture, loadManifest, syntheticManifest } from "./helpers";

/** A manifest as the worker wrote it before the textured model (schemaVersion 1.0). */
function v1Manifest(): SceneManifest {
  return { ...syntheticManifest(), schemaVersion: "1.0" } as unknown as SceneManifest;
}

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

  it("rejects a render mode or render asset other than the textured GLB", () => {
    const base = syntheticManifest();
    const issues = sceneManifestIssues({
      ...base,
      renderMode: "gaussian" as "mesh",
      assets: { ...base.assets, render: { ...base.assets.render, format: "ply" as "glb" } },
    });
    expect(issues).toEqual(
      expect.arrayContaining(["renderMode must be mesh", "render asset must be a GLB"]),
    );
  });

  it("rejects an unknown manifest schema version", () => {
    const future = { ...syntheticManifest(), schemaVersion: "3.0" as "2.0" };
    expect(sceneManifestIssues(future)).toEqual(["unsupported manifest schemaVersion"]);
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
      renderUrl: "/fixtures/scene.glb",
      collisionUrl: "/fixtures/collision.glb",
    });
    expect(r.renderFormat).toBe("glb");
    expect(r.collisionUrl).toBe("/fixtures/collision.glb");
    expect(() =>
      resolveViewerSource({
        kind: "urls",
        manifest: syntheticManifest(),
        renderUrl: "javascript:alert(1)",
      }),
    ).toThrow();
  });

  it("refuses a scene made before the textured model, with the user-facing message", () => {
    expect(OUTDATED_SCENE_MESSAGE).toBe(
      "This scene was made by an older version of Andarilho. Process the capture again to view it.",
    );
    const source = { kind: "urls" as const, manifest: v1Manifest(), renderUrl: "/fixtures/a.glb" };
    expect(() => resolveViewerSource(source)).toThrow(OutdatedSceneError);
    expect(() => resolveViewerSource(source)).toThrow(OUTDATED_SCENE_MESSAGE);
  });

  it("refuses an outdated delivery envelope with the same message", () => {
    const envelope: DeliveryEnvelope = {
      sceneRevisionId: "fixture-revision",
      manifestSha256: "0".repeat(64),
      expiresAt: new Date(Date.now() + 60_000).toISOString(),
      manifest: v1Manifest(),
      assetUrls: {
        render: {
          url: "https://cdn.example.test/scene.glb?sig=1",
          sha256: "0".repeat(64),
          bytes: 1,
          format: "glb",
        },
      },
    };
    expect(() => resolveViewerSource({ kind: "delivery", envelope })).toThrow(
      OUTDATED_SCENE_MESSAGE,
    );
  });

  it("reports a 1.0 manifest as an unsupported schema version to the checks", () => {
    expect(sceneManifestIssues(v1Manifest())).toEqual(["unsupported manifest schemaVersion"]);
  });

  it.skipIf(!hasFixture("synthetic-two-room.manifest.json"))("accepts the fixture manifest", () => {
    const manifest = loadManifest();
    expect(manifest.schemaVersion).toBe("2.0");
    expect(manifest.assets.render.key).toMatch(/model\.glb$/);
    expect(sceneManifestIssues(manifest)).toEqual([]);
    const resolved = resolveViewerSource({
      kind: "urls",
      manifest,
      renderUrl: "/fixtures/synthetic-two-room.model.glb",
    });
    expect(resolved.renderFormat).toBe("glb");
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
