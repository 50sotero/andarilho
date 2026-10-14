import { existsSync, readFileSync } from "node:fs";
import { resolve, sep } from "node:path";
import type { Layout, SceneManifest } from "@andarilho/scene-contract";

export const fixturesDir = resolve(import.meta.dirname, "../../scene-contract/fixtures") + sep;

export function fixturePath(name: string): string {
  return `${fixturesDir}${name}`;
}

export function hasFixture(name: string): boolean {
  return existsSync(fixturePath(name));
}

export function loadLayout(): Layout {
  return JSON.parse(readFileSync(fixturePath("synthetic-two-room.layout.json"), "utf8")) as Layout;
}

export function loadManifest(): SceneManifest {
  return JSON.parse(
    readFileSync(fixturePath("synthetic-two-room.manifest.json"), "utf8"),
  ) as SceneManifest;
}

/** Minimal valid manifest for the synthetic two-room layout (used when the fixture is absent). */
export function syntheticManifest(overrides: Partial<SceneManifest> = {}): SceneManifest {
  return {
    schemaVersion: "1.0",
    sceneId: "fixture-scene",
    sceneRevisionId: "fixture-revision",
    revision: 1,
    runId: "fixture-run",
    renderMode: "gaussian",
    units: "m",
    upAxis: "+Y",
    cameraForward: "-Z",
    scaleStatus: "user_measured",
    worldFromDecodedRender: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    geometrySpace: "world",
    bounds: { min: [-4.2, -0.1, -2.7], max: [4.2, 2.8, 2.7] },
    spawn: { feet: [2, 0, 0], yawRadians: Math.PI / 2, floorId: "floor-0" },
    floors: [{ id: "floor-0", elevation: 0 }],
    assets: {
      render: {
        key: "tenants/fixture-project/scenes/fixture-revision/scene.ply",
        format: "ply",
        bytes: 1,
        sha256: "0".repeat(64),
      },
      collision: {
        key: "tenants/fixture-project/scenes/fixture-revision/collision.glb",
        format: "glb",
        bytes: 1,
        sha256: "1".repeat(64),
      },
    },
    navigation: {
      mode: "approved_collision",
      layoutRevision: 1,
      approved: true,
      playerRadius: 0.25,
      playerHeight: 1.7,
      eyeHeight: 1.6,
    },
    quality: { registeredViews: 1, selectedViews: 1, disconnectedComponents: 1, warnings: [] },
    provenance: {
      recipeDigest: "fixture",
      captureSetId: "fixture",
      geometrySource: "synthetic_fixture",
    },
    ...overrides,
  };
}
