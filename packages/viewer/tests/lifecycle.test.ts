// @vitest-environment jsdom
import type { DeliveryEnvelope } from "@andarilho/scene-contract";
import {
  BoxGeometry,
  Group,
  Mesh,
  MeshStandardMaterial,
  Quaternion,
  Scene,
  type WebGLRenderer,
} from "three";
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
import { createMeshScene, type MeshSceneHandle } from "../src/render/mesh-scene";
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

/** A fake mesh scene: its load stays pending until the test settles it. */
interface MeshFake {
  handle: MeshSceneHandle;
  dispose: ReturnType<typeof vi.fn>;
  setMaxAnisotropy: ReturnType<typeof vi.fn>;
  initialized: ReturnType<typeof deferred<void>>;
}

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
  /** The first mesh scene the engine creates. */
  mesh: MeshFake;
  /** Every mesh scene, in creation order. */
  meshes: MeshFake[];
  createMeshScene: ReturnType<typeof vi.fn>;
  loop: { current: ((t: number) => void) | null };
}

/**
 * A mesh scene whose load stays pending until the test settles it. Disposing it
 * cancels the load and rejects `initialized`, as the real scene does.
 */
function fakeMeshScene(): MeshFake {
  const initialized = deferred<void>();
  // Nothing awaits a load cancelled after disposal, so its rejection is marked handled.
  initialized.promise.catch(() => undefined);
  const dispose = vi.fn(() =>
    initialized.reject(new DOMException("The scene load was cancelled", "AbortError")),
  );
  const setMaxAnisotropy = vi.fn();
  const handle: MeshSceneHandle = {
    root: null,
    initialized: initialized.promise,
    setMaxAnisotropy,
    dispose,
  };
  return { handle, dispose, setMaxAnisotropy, initialized };
}

/** A Spark scene whose load stays pending until the test settles it. */
interface SparkFake {
  handle: SparkSceneHandle;
  dispose: ReturnType<typeof vi.fn>;
  initialized: ReturnType<typeof deferred<void>>;
}

function fakeSparkScene(): SparkFake {
  const initialized = deferred<void>();
  initialized.promise.catch(() => undefined);
  const dispose = vi.fn();
  const handle = {
    splats: {},
    spark: {},
    initialized: initialized.promise,
    setSplatBudget: vi.fn(),
    dispose,
  } as unknown as SparkSceneHandle;
  return { handle, dispose, initialized };
}

function harness(spawn: SpawnValidation = { ok: true, feet: [2, 0, 0], supportY: 0 }): Harness {
  const canvas = document.createElement("canvas");
  const loop: Harness["loop"] = { current: null };
  const renderer = {
    domElement: canvas,
    capabilities: { getMaxAnisotropy: () => 16 },
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
  const meshes: MeshFake[] = [fakeMeshScene()];
  let meshCalls = 0;
  const createMeshScene = vi.fn(() => {
    // The Nth call gets the Nth scene, as the real factory creates a new one per load.
    meshCalls += 1;
    if (meshCalls > meshes.length) meshes.push(fakeMeshScene());
    return meshes[meshCalls - 1]!.handle;
  });
  const deps: Partial<ViewerDependencies> = {
    isWebGL2Available: () => true,
    createRenderer: () => renderer,
    createSparkScene,
    createMeshScene,
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
    mesh: meshes[0]!,
    meshes,
    world,
    observers,
    createSparkScene,
    createMeshScene,
    loop,
  };
}

const GLB_URL = "https://cdn.example.test/scene.glb?sig=1";

/** An envelope whose render asset is a textured GLB (the contract's format union omits "glb"). */
function glbEnvelope(): DeliveryEnvelope {
  const env = envelope();
  env.manifest.assets.render.format = "glb" as unknown as "ply";
  env.assetUrls.render = { url: GLB_URL, sha256: "0".repeat(64), bytes: 1, format: "glb" };
  return env;
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

  it("draws a GLB through the mesh scene rather than Spark", async () => {
    const h = harness();
    const { viewer } = start(h, { source: { kind: "delivery", envelope: glbEnvelope() } });
    await vi.waitFor(() => expect(h.createMeshScene).toHaveBeenCalled());
    expect(h.createSparkScene).not.toHaveBeenCalled();
    expect(h.createMeshScene).toHaveBeenCalledWith(
      expect.objectContaining({ url: GLB_URL, maxAnisotropy: 8 }),
    );

    const args = h.createMeshScene.mock.calls[0]![0] as { onProgress: (f: number | null) => void };
    args.onProgress(0.5);
    expect(viewer.getState()).toMatchObject({ progress: { renderFraction: 0.5 } });

    h.mesh.initialized.resolve();
    await vi.waitFor(() =>
      expect(viewer.getState()).toMatchObject({ status: "ready", mode: "walkable" }),
    );
    viewer.dispose();
  });

  it("applies quality changes to the mesh scene's anisotropy", async () => {
    const h = harness();
    const { viewer } = start(h, { source: { kind: "delivery", envelope: glbEnvelope() } });
    h.mesh.initialized.resolve();
    await vi.waitFor(() => expect(viewer.getState().status).toBe("ready"));
    viewer.setQuality("mobile");
    expect(h.mesh.setMaxAnisotropy).toHaveBeenLastCalledWith(4);
    viewer.setQuality("low");
    expect(h.mesh.setMaxAnisotropy).toHaveBeenLastCalledWith(1);
    expect(h.createSparkScene).not.toHaveBeenCalled();
    viewer.dispose();
  });

  it("applies a quality change made while the mesh scene is still being created", async () => {
    const h = harness();
    const created = deferred<MeshSceneHandle>();
    h.deps.createMeshScene = vi.fn(() => created.promise);
    const { viewer } = start(h, { source: { kind: "delivery", envelope: glbEnvelope() } });
    await vi.waitFor(() => expect(h.deps.createMeshScene).toHaveBeenCalledTimes(1));
    // No scene exists yet, so this change cannot reach it directly.
    viewer.setQuality("low");
    created.resolve(h.mesh.handle);
    await vi.waitFor(() => expect(h.mesh.setMaxAnisotropy).toHaveBeenLastCalledWith(1));
    viewer.dispose();
  });

  it("disposes the mesh scene with the viewer", async () => {
    const h = harness();
    const { viewer } = start(h, { source: { kind: "delivery", envelope: glbEnvelope() } });
    h.mesh.initialized.resolve();
    await vi.waitFor(() => expect(viewer.getState().status).toBe("ready"));
    viewer.dispose();
    expect(h.mesh.dispose).toHaveBeenCalledTimes(1);
    expect(h.spark.dispose).not.toHaveBeenCalled();
  });

  it("reloads the mesh scene after WebGL context loss", async () => {
    const h = harness();
    const { viewer } = start(h, { source: { kind: "delivery", envelope: glbEnvelope() } });
    h.mesh.initialized.resolve();
    await vi.waitFor(() => expect(viewer.getState().status).toBe("ready"));
    viewer.enterWalk();
    h.renderer.domElement.dispatchEvent(new Event("webglcontextlost", { cancelable: true }));
    h.renderer.domElement.dispatchEvent(new Event("webglcontextrestored"));
    await vi.waitFor(() => expect(h.createMeshScene).toHaveBeenCalledTimes(2));
    // The first scene is released; the reloaded one is the attached render.
    expect(h.meshes[0]!.dispose).toHaveBeenCalledTimes(1);
    expect(h.meshes[1]!.dispose).not.toHaveBeenCalled();
    h.meshes[1]!.initialized.resolve();
    await vi.waitFor(() =>
      expect(viewer.getState()).toMatchObject({ status: "ready", mode: "walkable" }),
    );
    expect(h.createSparkScene).not.toHaveBeenCalled();
    viewer.dispose();
    expect(h.meshes[1]!.dispose).toHaveBeenCalledTimes(1);
  });

  it("recovers when the context is lost while the first GLB is still downloading", async () => {
    const h = harness();
    const { viewer } = start(h, { source: { kind: "delivery", envelope: glbEnvelope() } });
    await vi.waitFor(() => expect(h.createMeshScene).toHaveBeenCalledTimes(1));
    h.renderer.domElement.dispatchEvent(new Event("webglcontextlost", { cancelable: true }));
    expect(viewer.getState().status).toBe("context-lost");
    h.renderer.domElement.dispatchEvent(new Event("webglcontextrestored"));
    // The recovery cancels the first download (its load rejects) and starts another.
    await vi.waitFor(() => expect(h.createMeshScene).toHaveBeenCalledTimes(2));
    expect(h.meshes[0]!.dispose).toHaveBeenCalledTimes(1);
    h.meshes[1]!.initialized.resolve();
    await vi.waitFor(() => expect(viewer.getState().status).toBe("ready"));
    expect(h.renderer.dispose).not.toHaveBeenCalled();
    viewer.dispose();
    expect(h.meshes[1]!.dispose).toHaveBeenCalledTimes(1);
  });

  it("recovers through the real mesh scene when the first download is cancelled", async () => {
    // Only the scene's own loader is real: its downloads wait for the test and reject
    // when aborted, as fetch does, so the cancellation takes the production path.
    const h = harness();
    const answers: ((response: Response) => void)[] = [];
    const download = ((_url: string, init?: RequestInit) =>
      new Promise<Response>((resolve, reject) => {
        answers.push(resolve);
        init?.signal?.addEventListener("abort", () =>
          reject(new DOMException("The download was aborted", "AbortError")),
        );
      })) as unknown as typeof fetch;
    h.deps.createMeshScene = (args) =>
      createMeshScene({
        renderer: args.renderer as WebGLRenderer,
        scene: args.scene,
        url: args.url,
        maxAnisotropy: args.maxAnisotropy,
        onProgress: args.onProgress,
        fetch: download,
        parse: async () => new Group(),
      });
    const { viewer } = start(h, { source: { kind: "delivery", envelope: glbEnvelope() } });
    await vi.waitFor(() => expect(answers).toHaveLength(1));

    h.renderer.domElement.dispatchEvent(new Event("webglcontextlost", { cancelable: true }));
    h.renderer.domElement.dispatchEvent(new Event("webglcontextrestored"));
    await vi.waitFor(() => expect(answers).toHaveLength(2));
    answers[1]!(new Response(new Uint8Array(8)));

    await vi.waitFor(() => expect(viewer.getState().status).toBe("ready"));
    viewer.dispose();
  });

  it("releases a mesh scene that arrives after a recovery, and keeps the recovered one", async () => {
    const h = harness();
    const late = fakeMeshScene();
    const recovered = fakeMeshScene();
    const third = fakeMeshScene();
    const pending = deferred<MeshSceneHandle>();
    // The first load is still creating its scene when the context is lost and restored.
    h.deps.createMeshScene = vi
      .fn()
      .mockReturnValueOnce(pending.promise)
      .mockReturnValueOnce(recovered.handle)
      .mockReturnValueOnce(third.handle);
    const { viewer } = start(h, { source: { kind: "delivery", envelope: glbEnvelope() } });
    await vi.waitFor(() => expect(h.deps.createMeshScene).toHaveBeenCalledTimes(1));
    const lose = () =>
      h.renderer.domElement.dispatchEvent(new Event("webglcontextlost", { cancelable: true }));
    const restore = () => h.renderer.domElement.dispatchEvent(new Event("webglcontextrestored"));

    lose();
    restore();
    await vi.waitFor(() => expect(h.deps.createMeshScene).toHaveBeenCalledTimes(2));
    recovered.initialized.resolve();
    await vi.waitFor(() => expect(viewer.getState().status).toBe("ready"));

    // The superseded load's scene finishes last: it must be released, not attached.
    pending.resolve(late.handle);
    await vi.waitFor(() => expect(late.dispose).toHaveBeenCalledTimes(1));
    expect(recovered.dispose).not.toHaveBeenCalled();
    expect(viewer.getState().status).toBe("ready");

    viewer.setQuality("low");
    expect(recovered.setMaxAnisotropy).toHaveBeenLastCalledWith(1);
    expect(late.setMaxAnisotropy).not.toHaveBeenCalled();

    // A second loss releases the recovered scene, not an orphan.
    lose();
    restore();
    await vi.waitFor(() => expect(h.deps.createMeshScene).toHaveBeenCalledTimes(3));
    expect(recovered.dispose).toHaveBeenCalledTimes(1);
    expect(late.dispose).toHaveBeenCalledTimes(1);
    viewer.dispose();
    expect(third.dispose).toHaveBeenCalledTimes(1);
  });

  it("releases a Spark scene that arrives after a recovery, and keeps the recovered one", async () => {
    const h = harness();
    const late = fakeSparkScene();
    const recovered = fakeSparkScene();
    const pending = deferred<SparkSceneHandle>();
    h.deps.createSparkScene = vi
      .fn()
      .mockReturnValueOnce(pending.promise)
      .mockReturnValueOnce(recovered.handle);
    const { viewer } = start(h);
    await vi.waitFor(() => expect(h.deps.createSparkScene).toHaveBeenCalledTimes(1));
    h.renderer.domElement.dispatchEvent(new Event("webglcontextlost", { cancelable: true }));
    h.renderer.domElement.dispatchEvent(new Event("webglcontextrestored"));
    await vi.waitFor(() => expect(h.deps.createSparkScene).toHaveBeenCalledTimes(2));
    recovered.initialized.resolve();
    await vi.waitFor(() => expect(viewer.getState().status).toBe("ready"));

    pending.resolve(late.handle);
    await vi.waitFor(() => expect(late.dispose).toHaveBeenCalledTimes(1));
    expect(recovered.dispose).not.toHaveBeenCalled();
    viewer.setQuality("low");
    expect(recovered.handle.setSplatBudget).toHaveBeenLastCalledWith(400_000);
    expect(late.handle.setSplatBudget).not.toHaveBeenCalled();
    viewer.dispose();
    expect(recovered.dispose).toHaveBeenCalledTimes(1);
  });

  it("does not attach a GLB's transform from the manifest", async () => {
    // The textured GLB is already world space with identity nodes, so the manifest's
    // render transform (for the splat formats) must not move it. The writer contract is
    // not in this repo yet; this pins the viewer side.
    const h = harness();
    const scene = new Scene();
    const cube = new Group();
    cube.add(new Mesh(new BoxGeometry(), new MeshStandardMaterial()));
    h.deps.createMeshScene = (args) =>
      createMeshScene({
        renderer: args.renderer as WebGLRenderer,
        scene,
        url: args.url,
        maxAnisotropy: args.maxAnisotropy,
        onProgress: args.onProgress,
        fetch: (async () => new Response(new Uint8Array(8))) as unknown as typeof fetch,
        parse: async () => cube,
      });
    const env = glbEnvelope();
    // A 90 degree turn about +Y, plus a 2x scale: valid for splats, wrong for the GLB.
    env.manifest.worldFromDecodedRender = [0, 0, -2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1];
    const { viewer } = start(h, { source: { kind: "delivery", envelope: env } });
    await vi.waitFor(() => expect(viewer.getState().status).toBe("ready"));

    expect(scene.children).toContain(cube);
    expect(cube.position.toArray()).toEqual([0, 0, 0]);
    expect(cube.quaternion.equals(new Quaternion())).toBe(true);
    expect(cube.scale.toArray()).toEqual([1, 1, 1]);
    viewer.dispose();
  });

  it("recovers from a second context loss during the first reload", async () => {
    const h = harness();
    const { viewer } = start(h, { source: { kind: "delivery", envelope: glbEnvelope() } });
    h.mesh.initialized.resolve();
    await vi.waitFor(() => expect(viewer.getState().status).toBe("ready"));
    const lose = () =>
      h.renderer.domElement.dispatchEvent(new Event("webglcontextlost", { cancelable: true }));
    const restore = () => h.renderer.domElement.dispatchEvent(new Event("webglcontextrestored"));

    lose();
    restore();
    await vi.waitFor(() => expect(h.createMeshScene).toHaveBeenCalledTimes(2));
    // The reload is still downloading when the context is lost again.
    lose();
    restore();
    await vi.waitFor(() => expect(h.createMeshScene).toHaveBeenCalledTimes(3));
    expect(h.meshes[1]!.dispose).toHaveBeenCalledTimes(1);
    h.meshes[2]!.initialized.resolve();
    await vi.waitFor(() => expect(viewer.getState().status).toBe("ready"));
    expect(h.createMeshScene).toHaveBeenCalledTimes(3);
    expect(h.meshes[2]!.dispose).not.toHaveBeenCalled();
    viewer.dispose();
  });
});
