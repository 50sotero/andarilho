// @vitest-environment jsdom
import type { DeliveryEnvelope } from "@andarilho/scene-contract";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  createViewer,
  type CollisionWorldLike,
  type CreateViewerOptions,
  type RendererLike,
  type ViewerDependencies,
} from "../src/create-viewer";
import type { RapierModule, SpawnValidation } from "../src/movement/collision";
import { writeCollisionGlb } from "../src/movement/glb";
import type { SparkSceneHandle } from "../src/render/spark-scene";
import { layoutToCollisionMesh } from "../src/review/floorplan";
import type { Vec3 } from "../src/types";
import type { ViewerState } from "../src/viewer-state";
import { loadLayout, syntheticManifest } from "./helpers";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const glbBytes = writeCollisionGlb(layoutToCollisionMesh(loadLayout()));

interface Harness {
  deps: Partial<ViewerDependencies>;
  renderer: RendererLike & {
    setAnimationLoop: ReturnType<typeof vi.fn>;
    dispose: ReturnType<typeof vi.fn>;
    render: ReturnType<typeof vi.fn>;
  };
  spark: {
    handle: SparkSceneHandle;
    dispose: ReturnType<typeof vi.fn>;
    initialized: ReturnType<typeof deferred<void>>;
  };
  world: CollisionWorldLike & { dispose: ReturnType<typeof vi.fn> };
  observers: { observe: ReturnType<typeof vi.fn>; disconnect: ReturnType<typeof vi.fn> }[];
  createSparkScene: ReturnType<typeof vi.fn>;
  loop: { current: ((t: number) => void) | null };
}

function harness(spawn: SpawnValidation = { ok: true, feet: [2, 0, 0], supportY: 0 }): Harness {
  const canvas = document.createElement("canvas");
  const loop: Harness["loop"] = { current: null };
  const renderer = {
    domElement: canvas,
    setAnimationLoop: vi.fn((cb: ((t: number) => void) | null) => {
      loop.current = cb;
    }),
    setPixelRatio: vi.fn(),
    setSize: vi.fn(),
    render: vi.fn(),
    dispose: vi.fn(),
  };
  const initialized = deferred<void>();
  const sparkDispose = vi.fn();
  const handle = {
    splats: {},
    spark: {},
    initialized: initialized.promise,
    setSplatBudget: vi.fn(),
    dispose: sparkDispose,
  } as unknown as SparkSceneHandle;
  let feet: Vec3 = [2, 0, 0];
  const world = {
    validateSpawn: vi.fn(() => spawn),
    setSpawn: vi.fn((f: Vec3) => {
      feet = [...f];
    }),
    resetToSpawn: vi.fn(),
    step: vi.fn(() => ({ feet, grounded: true, recovered: false })),
    state: {
      get feet() {
        return feet;
      },
      interpolatedFeet: () => feet,
    },
    dispose: vi.fn(),
  };
  const observers: Harness["observers"] = [];
  class FakeResizeObserver {
    observe = vi.fn();
    unobserve = vi.fn();
    disconnect = vi.fn();
    constructor() {
      observers.push(this);
    }
  }
  const createSparkScene = vi.fn(() => handle);
  const deps: Partial<ViewerDependencies> = {
    isWebGL2Available: () => true,
    createRenderer: () => renderer,
    createSparkScene,
    initRapier: async () => ({}) as RapierModule,
    createCollisionWorld: () => world,
    fetchBytes: async () => glbBytes.slice().buffer,
    now: () => performance.now(),
    ResizeObserver: FakeResizeObserver as unknown as typeof ResizeObserver,
  };
  return {
    deps,
    renderer,
    spark: { handle, dispose: sparkDispose, initialized },
    world,
    observers,
    createSparkScene,
    loop,
  };
}

function envelope(): DeliveryEnvelope {
  const manifest = syntheticManifest();
  return {
    sceneRevisionId: manifest.sceneRevisionId,
    manifestSha256: "0".repeat(64),
    expiresAt: new Date(Date.now() + 300_000).toISOString(),
    manifest,
    assetUrls: {
      render: {
        url: "https://cdn.example.test/scene.ply?sig=1",
        sha256: "0".repeat(64),
        bytes: 1,
        format: "ply",
      },
      collision: {
        url: "https://cdn.example.test/collision.glb?sig=1",
        sha256: "",
        bytes: 1,
        format: "glb",
      },
    },
  };
}

let container: HTMLDivElement;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
});

afterEach(() => {
  container.remove();
  vi.restoreAllMocks();
});

function start(h: Harness, extra: Partial<CreateViewerOptions> = {}) {
  const states: ViewerState[] = [];
  const viewer = createViewer({
    container,
    source: { kind: "delivery", envelope: envelope() },
    quality: "desktop",
    dependencies: h.deps,
    onStateChange: (s) => states.push(s),
    ...extra,
  });
  return { viewer, states };
}

describe("viewer lifecycle", () => {
  it("enables walking only after render, collision, physics and spawn succeed", async () => {
    const h = harness();
    const { viewer, states } = start(h);
    await vi.waitFor(() => expect(h.createSparkScene).toHaveBeenCalled());
    await vi.waitFor(() => expect(h.world.setSpawn).toHaveBeenCalled());
    expect(viewer.getState().status).toBe("loading");
    expect(viewer.enterWalk()).toBe(false);

    h.spark.initialized.resolve();
    await vi.waitFor(() =>
      expect(viewer.getState()).toMatchObject({ status: "ready", mode: "walkable" }),
    );
    expect(states.map((s) => s.status)).toContain("loading");
    expect(container.contains(h.renderer.domElement)).toBe(true);
    expect(h.renderer.setAnimationLoop).toHaveBeenCalledTimes(1);
    expect(h.observers[0]!.observe).toHaveBeenCalledWith(container);

    expect(viewer.enterWalk()).toBe(true);
    expect(viewer.getState().status).toBe("walking");
    h.loop.current?.(0);
    h.loop.current?.(16);
    expect(h.renderer.render).toHaveBeenCalled();
    viewer.exitWalk();
    expect(viewer.getState()).toMatchObject({ status: "ready", mode: "walkable" });
    viewer.dispose();
  });

  it("releases every resource on dispose", async () => {
    const h = harness();
    const removeDoc = vi.spyOn(document, "removeEventListener");
    const removeWin = vi.spyOn(window, "removeEventListener");
    const { viewer } = start(h);
    h.spark.initialized.resolve();
    await vi.waitFor(() => expect(viewer.getState().status).toBe("ready"));
    viewer.setCollisionOverlay(true);
    const listener = vi.fn();
    viewer.subscribe(listener);

    viewer.dispose();
    expect(h.renderer.setAnimationLoop).toHaveBeenLastCalledWith(null);
    expect(h.renderer.dispose).toHaveBeenCalledTimes(1);
    expect(h.spark.dispose).toHaveBeenCalledTimes(1);
    expect(h.world.dispose).toHaveBeenCalled();
    expect(h.observers[0]!.disconnect).toHaveBeenCalled();
    expect(container.contains(h.renderer.domElement)).toBe(false);
    expect(removeDoc.mock.calls.map((c) => c[0])).toEqual(
      expect.arrayContaining(["visibilitychange", "pointerlockchange"]),
    );
    expect(removeWin.mock.calls.map((c) => c[0])).toEqual(
      expect.arrayContaining(["keydown", "keyup", "blur"]),
    );
    expect(viewer.getState().status).toBe("disposed");
    expect(listener).toHaveBeenCalledWith({ status: "disposed" });

    // Idempotent and inert afterwards.
    viewer.dispose();
    expect(h.renderer.dispose).toHaveBeenCalledTimes(1);
    expect(viewer.enterWalk()).toBe(false);
  });

  it("ignores a render load that completes after dispose", async () => {
    const h = harness();
    const { viewer, states } = start(h);
    await vi.waitFor(() => expect(h.createSparkScene).toHaveBeenCalled());
    viewer.dispose();
    expect(h.spark.dispose).toHaveBeenCalledTimes(1);
    h.spark.initialized.resolve();
    await new Promise((r) => setTimeout(r, 10));
    expect(viewer.getState().status).toBe("disposed");
    expect(states.some((s) => s.status === "ready")).toBe(false);
    expect(states.at(-1)?.status).toBe("disposed");
  });

  it("releases a Spark scene created after dispose", async () => {
    const h = harness();
    const pending = deferred<SparkSceneHandle>();
    h.deps.createSparkScene = vi.fn(() => pending.promise);
    const { viewer } = start(h);
    await vi.waitFor(() => expect(h.deps.createSparkScene).toHaveBeenCalled());
    viewer.dispose();
    pending.resolve(h.spark.handle);
    await vi.waitFor(() => expect(h.spark.dispose).toHaveBeenCalledTimes(1));
    expect(viewer.getState().status).toBe("disposed");
  });

  it("releases a collision world created after dispose", async () => {
    const h = harness();
    const rapier = deferred<RapierModule>();
    h.deps.initRapier = () => rapier.promise;
    const { viewer } = start(h);
    await vi.waitFor(() => expect(h.createSparkScene).toHaveBeenCalled());
    viewer.dispose();
    rapier.resolve({} as RapierModule);
    await new Promise((r) => setTimeout(r, 10));
    expect(h.world.setSpawn).not.toHaveBeenCalled();
    expect(viewer.getState().status).toBe("disposed");
  });

  it("disposes partially created objects when initialization fails", async () => {
    const h = harness();
    h.deps.ResizeObserver = class {
      constructor() {
        throw new Error("boom");
      }
    } as unknown as typeof ResizeObserver;
    const { viewer } = start(h);
    await vi.waitFor(() => expect(viewer.getState().status).toBe("error"));
    expect(viewer.getState()).toMatchObject({ error: { code: "init-failed", recoverable: true } });
    expect(h.renderer.dispose).toHaveBeenCalledTimes(1);
    expect(h.renderer.setAnimationLoop).not.toHaveBeenCalledWith(expect.any(Function));
    expect(container.contains(h.renderer.domElement)).toBe(false);
    viewer.dispose();
  });

  it("reports a recoverable error when the render asset fails", async () => {
    const h = harness();
    const { viewer } = start(h);
    h.spark.initialized.reject(new Error("HTTP 403 https://secret.example/signed"));
    await vi.waitFor(() => expect(viewer.getState().status).toBe("error"));
    const state = viewer.getState();
    expect(state).toMatchObject({ error: { code: "render-load-failed", recoverable: true } });
    expect(JSON.stringify(state)).not.toContain("secret.example");
    // The failed engine stops rendering and frees the GPU before the page unmounts it.
    expect(h.spark.dispose).toHaveBeenCalledTimes(1);
    expect(h.renderer.dispose).toHaveBeenCalledTimes(1);
    expect(h.renderer.setAnimationLoop).toHaveBeenLastCalledWith(null);
    viewer.dispose();
    expect(h.spark.dispose).toHaveBeenCalledTimes(1);
  });

  it("falls back to inspection when collision is missing or invalid", async () => {
    const h = harness();
    const env = envelope();
    delete env.assetUrls.collision;
    const { viewer } = start(h, { source: { kind: "delivery", envelope: env } });
    h.spark.initialized.resolve();
    await vi.waitFor(() =>
      expect(viewer.getState()).toMatchObject({ status: "ready", mode: "inspection" }),
    );
    expect(viewer.enterWalk()).toBe(false);
    viewer.dispose();

    const h2 = harness();
    h2.deps.fetchBytes = async () =>
      writeCollisionGlb(layoutToCollisionMesh(loadLayout()), { scale: [2, 2, 2] })
        .buffer as ArrayBuffer;
    const second = start(h2);
    h2.spark.initialized.resolve();
    await vi.waitFor(() =>
      expect(second.viewer.getState()).toMatchObject({
        status: "ready",
        mode: "inspection",
        inspectionReason: "The collision asset is invalid.",
      }),
    );
    second.viewer.dispose();
  });

  it("reports a failed collision download as recoverable, not as inspection-only", async () => {
    const h = harness();
    h.deps.fetchBytes = async () => {
      throw new Error("Collision asset request failed (403)"); // e.g. an expired signed URL
    };
    const { viewer } = start(h);
    h.spark.initialized.resolve();
    await vi.waitFor(() =>
      expect(viewer.getState()).toMatchObject({
        status: "error",
        error: { code: "collision-load-failed", recoverable: true },
      }),
    );
    expect(viewer.enterWalk()).toBe(false);
    expect(h.renderer.setAnimationLoop).toHaveBeenLastCalledWith(null);
    expect(h.renderer.dispose).toHaveBeenCalledTimes(1);
    expect(h.spark.dispose).toHaveBeenCalledTimes(1);
    expect(container.contains(h.renderer.domElement)).toBe(false);
    h.renderer.render.mockClear();
    h.loop.current?.(100);
    expect(h.renderer.render).not.toHaveBeenCalled();
    viewer.dispose();
  });

  it("falls back to inspection when the spawn has no clearance", async () => {
    const h = harness({ ok: false, reason: "blocked", supportY: 0 });
    const { viewer } = start(h);
    h.spark.initialized.resolve();
    await vi.waitFor(() =>
      expect(viewer.getState()).toMatchObject({ status: "ready", mode: "inspection" }),
    );
    expect(h.world.dispose).toHaveBeenCalled();
    viewer.dispose();
  });

  it("refuses to start without WebGL2 and creates nothing", async () => {
    const h = harness();
    const createRenderer = vi.fn(() => h.renderer);
    h.deps.createRenderer = createRenderer;
    h.deps.isWebGL2Available = () => false;
    const { viewer } = start(h);
    await vi.waitFor(() =>
      expect(viewer.getState()).toMatchObject({
        status: "error",
        error: { code: "webgl2-unavailable" },
      }),
    );
    expect(createRenderer).not.toHaveBeenCalled();
    viewer.dispose();
  });

  it("rejects an expired or inconsistent envelope", async () => {
    const h = harness();
    const env = envelope();
    env.expiresAt = new Date(Date.now() - 1000).toISOString();
    const { viewer } = start(h, { source: { kind: "delivery", envelope: env } });
    await vi.waitFor(() =>
      expect(viewer.getState()).toMatchObject({
        status: "error",
        error: { code: "invalid-input" },
      }),
    );
    viewer.dispose();
  });

  it("recovers from WebGL context loss", async () => {
    const h = harness();
    const { viewer } = start(h);
    h.spark.initialized.resolve();
    await vi.waitFor(() => expect(viewer.getState().status).toBe("ready"));
    viewer.enterWalk();
    const lost = new Event("webglcontextlost", { cancelable: true });
    h.renderer.domElement.dispatchEvent(lost);
    expect(lost.defaultPrevented).toBe(true);
    expect(viewer.getState()).toMatchObject({ status: "context-lost", resumeMode: "walkable" });
    h.renderer.render.mockClear();
    h.loop.current?.(100);
    expect(h.renderer.render).not.toHaveBeenCalled();

    h.renderer.domElement.dispatchEvent(new Event("webglcontextrestored"));
    await vi.waitFor(() =>
      expect(viewer.getState()).toMatchObject({ status: "ready", mode: "walkable" }),
    );
    expect(h.createSparkScene).toHaveBeenCalledTimes(2);
    expect(h.spark.dispose).toHaveBeenCalledTimes(1);
    viewer.dispose();
  });

  it("applies quality changes through the handle", async () => {
    const h = harness();
    const { viewer } = start(h);
    h.spark.initialized.resolve();
    await vi.waitFor(() => expect(viewer.getState().status).toBe("ready"));
    viewer.setQuality("low");
    expect(viewer.getState()).toMatchObject({ quality: "low" });
    expect(h.spark.handle.setSplatBudget).toHaveBeenCalledWith(400_000);
    expect(viewer.getDiagnostics().qualitySetting).toBe("low");
    viewer.dispose();
  });
});
